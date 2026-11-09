# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
import sys
from unittest.mock import Mock

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import check_accuracy, skip_no_hopper

import tensorrt_llm._torch.modules.triton_linear as triton_linear_module
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mxfp8_utils import (dequant_mxfp8_weight,
                                                     quant_bf16_to_mxfp8)
from tensorrt_llm._torch.modules.triton_linear import TritonLinear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@pytest.mark.parametrize(
    "linear_cls",
    [Linear, pytest.param(TritonLinear, marks=skip_no_hopper)])
def test_linear_unquantized(linear_cls):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    num_tokens = 128
    hidden_size = 64
    out_size = 256
    dtype = torch.bfloat16
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    w = torch.randn((hidden_size, out_size), dtype=dtype).cuda()
    b = torch.randn((out_size, ), dtype=dtype).cuda()

    weights = {
        "weight": w.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
    }

    linear = linear_cls(
        in_features=hidden_size,
        out_features=out_size,
        bias=True,
        dtype=dtype,
    )
    linear.load_weights([weights])
    linear.cuda()

    actual_c = linear.forward(x)
    reference_c = torch.matmul(x, w) + b

    check_accuracy(actual_c, reference_c, atol=0.01, rtol=0.01, percent=0.99)


@pytest.mark.parametrize(
    "linear_cls",
    [Linear, pytest.param(TritonLinear, marks=skip_no_hopper)])
def test_linear_fp8qdq(linear_cls):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    num_tokens = 128
    hidden_size = 64
    out_size = 256
    dtype = torch.bfloat16
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    qx, sx = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    w = torch.randn((hidden_size, out_size), dtype=dtype).cuda()
    qw, sw = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(w)
    b = torch.randn((out_size, ), dtype=dtype).cuda()

    weights = {
        "weight": qw.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
        "input_scale": sx,
        "weight_scale": sw,
    }

    linear = linear_cls(in_features=hidden_size,
                        out_features=out_size,
                        bias=True,
                        dtype=dtype,
                        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8))
    linear.load_weights([weights])
    linear.cuda()

    actual_c = linear.forward(qx)
    x_qdq = qx.to(torch.float32) * sx
    w_qdq = qw.to(torch.float32) * sw
    reference_c = torch.matmul(x_qdq, w_qdq) + b

    check_accuracy(actual_c,
                   reference_c.to(dtype),
                   atol=0.01,
                   rtol=0.01,
                   percent=0.99)


@skip_no_hopper
@pytest.mark.parametrize(
    ("num_tokens", "in_size", "out_size", "has_bias"),
    [
        (1, 6144, 9216, False),  # M3.1 fused qkv at TP1
        (16, 3072, 6144, False),
        (64, 6144, 256, True),
        (1024, 256, 64, False),
        (8, 96, 64, False),  # K not a multiple of block_k
        (128, 64, 6144, False),
    ],
)
def test_linear_mxfp8_triton_graph_matches_fp32(
    num_tokens: int,
    in_size: int,
    out_size: int,
    has_bias: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A captured Hopper forward reads the MXFP8 weight in place and matches FP32 to BF16 precision."""
    dtype = torch.bfloat16
    torch.manual_seed(0)
    weight = torch.randn((out_size, in_size), dtype=torch.float32)
    # Vary the exponent between 1x32 blocks, as real weights do.
    exponents = torch.randint(-4,
                              5, (out_size, in_size // 32),
                              dtype=torch.float32)
    weight *= torch.exp2(exponents).repeat_interleave(32, 1)
    weight_e4m3, scale = quant_bf16_to_mxfp8(weight, 32)
    bias = torch.randn((out_size, ), dtype=dtype) if has_bias else None
    weights = {"weight": weight_e4m3, "weight_scale_inv": scale}
    if has_bias:
        weights["bias"] = bias

    linear = Linear(
        in_features=in_size,
        out_features=out_size,
        bias=has_bias,
        dtype=dtype,
        quant_config=QuantConfig(quant_algo=QuantAlgo.MXFP8, group_size=32),
    ).cuda()
    linear.load_weights([weights])

    assert linear.quant_method.use_triton
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.weight_scale.shape == (out_size, in_size // 32)

    x = torch.randn((num_tokens, in_size), dtype=dtype).cuda()
    weight_fp32 = dequant_mxfp8_weight(weight_e4m3, scale, 32).float().cuda()
    reference = x.float() @ weight_fp32.T
    if has_bias:
        reference += bias.float().cuda()

    # Fail if the captured forward falls back to the dequantizing path.
    def no_dequant(*args: object, **kwargs: object) -> None:
        raise AssertionError("captured forward must not dequantize the weight")

    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.mxfp8_utils.dequant_mxfp8_weight",
        no_dequant)
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            output = linear(x)
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()

    assert output.dtype == dtype
    relative_error = (output.float() - reference).norm() / reference.norm()
    assert relative_error < 5e-3, f"relative error {relative_error}"


@skip_no_hopper
@pytest.mark.parametrize("activation_dtype",
                         [torch.bfloat16, torch.float8_e4m3fn])
def test_linear_mxfp4(activation_dtype, monkeypatch):
    if activation_dtype == torch.float8_e4m3fn:
        pytest.skip("Latest Triton requires BF16 activation on Hopper")

    dtype = torch.bfloat16
    num_tokens = 128
    hidden_size = 256  # Must be even and divisible by 32 for MXFP4
    out_size = 512
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    w = torch.randn((hidden_size, out_size), dtype=dtype).cuda()
    b = torch.randn((out_size, ), dtype=dtype).cuda()

    from triton_kernels.numerics_details.mxfp import (downcast_to_mxfp_torch,
                                                      upcast_from_mxfp_torch)

    def fp32_to_mxfp4(tensor):
        # tensor (in_features, out_features)
        tensor = tensor.unsqueeze(0)
        tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor,
                                                           torch.uint8,
                                                           axis=1)
        return tensor_fp4[0], tensor_scales[0]

    def mxfp4_to_fp32(tensor, scales):
        tensor = tensor.unsqueeze(0)
        scales = scales.unsqueeze(0)
        tensor = upcast_from_mxfp_torch(tensor, scales, torch.float32, axis=1)
        return tensor[0]

    # Convert weight to MXFP4
    w_weight_fp4, w_weight_scale = fp32_to_mxfp4(w)
    w_weight_qdq = mxfp4_to_fp32(w_weight_fp4, w_weight_scale)

    # Create reference linear with dequantized weights
    ref_weights = {
        "weight": w_weight_qdq.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
    }

    ref_linear = Linear(  # Always use regular Linear for reference
        in_features=hidden_size,
        out_features=out_size,
        bias=True,
        dtype=dtype,
    )
    ref_linear.load_weights([ref_weights])
    ref_linear.cuda()

    ref_output = ref_linear.forward(x)
    torch.cuda.synchronize()

    # Now test with MXFP4 quantized weights
    weights = {
        "weight": w_weight_fp4.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
        "weight_scale":
        w_weight_scale.T,  # Transpose scale to match weight shape
    }

    # Add input scale for FP8 activation
    if activation_dtype == torch.float8_e4m3fn:
        _, input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
        weights["input_scale"] = input_scale

    quant_algo = QuantAlgo.W4A8_MXFP4_FP8 if activation_dtype == torch.float8_e4m3fn else QuantAlgo.W4A16_MXFP4

    linear = TritonLinear(in_features=hidden_size,
                          out_features=out_size,
                          bias=True,
                          dtype=dtype,
                          quant_config=QuantConfig(quant_algo=quant_algo))
    linear.load_weights([weights])
    linear.cuda()

    output = linear.forward(x)
    torch.cuda.synchronize()

    # Compare outputs with more relaxed tolerance for MXFP4
    check_accuracy(output, ref_output, rtol=0.2, atol=0.2, percent=0.95)

    matmul_spy = Mock(wraps=triton_linear_module.matmul)
    monkeypatch.setattr(triton_linear_module, "matmul", matmul_spy)
    method = triton_linear_module.TritonMXFP4LinearMethod(activation_dtype)
    direct_output = method.apply(linear, x, linear.bias)

    matmul_spy.assert_called_once()
    precision_config = matmul_spy.call_args.kwargs["precision_config"]
    assert precision_config.b_microblock_size == (
        triton_linear_module.MXFP_BLOCK_SIZE.value) == 32
    assert precision_config.b_mx_scale is linear.weight_scale
    check_accuracy(direct_output, ref_output, rtol=0.2, atol=0.2, percent=0.95)
