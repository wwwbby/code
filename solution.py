"""
HiF4 solution.py 提交接口模板
==============================

本文件只说明参赛者需要实现的 6 个公开函数，以及这些函数的输入/输出数据契约。
不包含任何 HiF4 参考量化或标准量化实现。

必须实现的 6 个函数：

Linear:
    1. hif4_calibration_and_quantize_weight
    2. hif4_dynamic_quantize_activation

Attention:
    3. hif4_calibration_attention
    4. hif4_dynamic_quantize_q
    5. hif4_dynamic_quantize_k
    6. hif4_dynamic_quantize_v

说明：
- 输入的 Weight / Activation / Q / K / V 均以 NVFP4 carrier + block scale 的形式提供。
- calibration 函数可以根据校准数据生成后续动态量化需要使用的 state。
- dynamic 函数通过对应 state 获取 calibration 阶段产生的固定信息。
- 选手自行实现 HiF4 量化算法；本模板不会提供任何 HiF4 参考实现。
"""

from __future__ import annotations

import math
from typing import Any

import torch


# =============================================================================
# NVFP4 helper
# =============================================================================

def dequantize_nvfp4(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    """将接口中的 NVFP4 carrier 还原为 BF16 Tensor。

    Args:
        quant_float:
            NVFP4 value carrier，shape 为 ``(..., C)``。

        scale_float:
            NVFP4 block scale，shape 为 ``(..., C // blk_size)``。

        blk_size:
            NVFP4 block size，默认为 16。

    Returns:
        BF16 Tensor，shape 与 ``quant_float`` 相同。
    """
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )

    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


# =============================================================================
# HiF4 demo quantizer
# =============================================================================

def _build_e6m2_values() -> tuple[float, ...]:
    """Return all finite positive E6M2 values accepted by the checker."""
    values: list[float] = []
    for exponent in range(-48, 16):
        for mantissa_code in range(4):
            value = (2.0 ** exponent) * (1.0 + 0.25 * mantissa_code)
            if value <= 49152.0:  # The remaining top code is NaN.
                values.append(float(value))
    return tuple(values)


_E6M2_VALUES = _build_e6m2_values()
_E6M2_TABLE_CPU = torch.tensor(_E6M2_VALUES, dtype=torch.float32)


def _deterministic_signs(length: int, seed: int) -> torch.Tensor:
    """Build a reproducible CPU Rademacher vector without touching global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(
        0,
        2,
        (length,),
        generator=generator,
        dtype=torch.int8,
    )
    return signs.to(torch.float32).mul_(2.0).sub_(1.0)


def _apply_hadamard_rotation(
    x: torch.Tensor,
    signs: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Apply the block-diagonal orthogonal transform D @ H to the last axis.

    Small blocks intentionally align with the HiF4 micro-scale tree: H4 stays
    inside one level-3 group and H8 stays inside one level-2 group.  Applying
    the same transform to both sides of a matmul leaves the FP32 result intact.
    """
    shape = tuple(int(v) for v in x.shape)
    channels = int(shape[-1])
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError(f"Hadamard block size must be a power of two, got {block_size}")
    if channels % block_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by rotation block {block_size}"
        )
    if signs.numel() != channels:
        raise ValueError(
            f"rotation has {signs.numel()} signs, expected {channels}"
        )

    blocks = x.to(torch.float32).reshape(-1, channels // block_size, block_size)
    blocks = blocks * signs.to(device=blocks.device, dtype=torch.float32).reshape(
        1, channels // block_size, block_size
    )

    stride = 1
    while stride < block_size:
        pairs = blocks.reshape(
            int(blocks.shape[0]),
            int(blocks.shape[1]),
            block_size // (2 * stride),
            2,
            stride,
        )
        left = pairs[..., 0, :]
        right = pairs[..., 1, :]
        blocks = torch.cat((left + right, left - right), dim=-1).reshape(
            int(blocks.shape[0]),
            int(blocks.shape[1]),
            block_size,
        )
        stride *= 2

    blocks = blocks * (block_size ** -0.5)
    return blocks.reshape(shape)


def _e6m2_table(device: torch.device) -> torch.Tensor:
    if device.type == "cpu":
        return _E6M2_TABLE_CPU
    return _E6M2_TABLE_CPU.to(device=device)


def _nearest_e6m2_index(
    target: torch.Tensor,
    table: torch.Tensor,
) -> torch.Tensor:
    """Find the closest legal finite E6M2 value for each positive target."""
    target = target.to(torch.float32).clamp(
        min=float(_E6M2_VALUES[0]),
        max=float(_E6M2_VALUES[-1]),
    )
    hi = torch.searchsorted(table, target).clamp(max=table.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    lo_value = table[lo]
    hi_value = table[hi]
    return torch.where(target - lo_value <= hi_value - target, lo, hi)


def _pack_hif4(
    original_shape: tuple[int, ...],
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
    sign: torch.Tensor,
    mant: torch.Tensor,
) -> dict[str, torch.Tensor]:
    channels = int(original_shape[-1])
    prefix = original_shape[:-1] + (channels // 64,)
    return {
        "scale_factor": scale_factor.reshape(prefix + (1, 1, 1)),
        "scale_lv2": scale_lv2.reshape(prefix + (8, 1, 1)),
        "scale_lv3": scale_lv3.reshape(prefix + (8, 2, 1)),
        "sign": sign.reshape(prefix + (8, 2, 4)),
        "mant": mant.reshape(prefix + (8, 2, 4)),
    }


def _encode_s1p2(
    blocks: torch.Tensor,
    scale_factor: torch.Tensor,
    scale_lv2: torch.Tensor,
    scale_lv3: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    divisor = (
        scale_factor[:, None, None, None]
        * scale_lv2[:, :, None, None]
        * scale_lv3[:, :, :, None]
    )
    scaled = blocks / divisor
    mant = torch.round(scaled.abs() * 4.0).clamp_(0.0, 7.0) * 0.25
    sign = torch.sign(scaled)
    sign = torch.where(mant == 0.0, torch.zeros_like(sign), sign)
    return sign.to(torch.float32), mant.to(torch.float32)


def _quantize_hif4_direct(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Paper-style peak-based BF16/FP32 to HiF4 conversion baseline."""
    original_shape = tuple(int(v) for v in x.shape)
    channels = int(original_shape[-1])
    if channels % 64 != 0:
        raise ValueError(f"last dimension {channels} is not divisible by 64")

    blocks = x.to(torch.float32).reshape(-1, 8, 2, 4)
    table = _e6m2_table(blocks.device)
    max4 = blocks.abs().amax(dim=-1)
    max8 = max4.amax(dim=-1)
    max64 = max8.amax(dim=-1)
    sf_index = _nearest_e6m2_index(max64 / 7.0, table)
    scale_factor = table[sf_index]

    scale_lv2 = torch.where(
        max8 / scale_factor[:, None] >= 4.0,
        2.0,
        1.0,
    ).to(torch.float32)
    scale_lv3 = torch.where(
        max4 / (scale_factor[:, None, None] * scale_lv2[:, :, None]) >= 2.0,
        2.0,
        1.0,
    ).to(torch.float32)
    sign, mant = _encode_s1p2(
        blocks,
        scale_factor,
        scale_lv2,
        scale_lv3,
    )
    return _pack_hif4(
        original_shape,
        scale_factor,
        scale_lv2,
        scale_lv3,
        sign,
        mant,
    )


def _search_chunk(
    blocks: torch.Tensor,
    table: torch.Tensor,
    radius: int,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimize elementwise SSE under the legal two-level micro-scale tree."""
    peak = blocks.abs().amax(dim=(1, 2, 3))
    base_index = _nearest_e6m2_index(peak / 7.0, table)
    offsets = torch.arange(-radius, radius + 1, device=blocks.device)
    candidate_index = (
        base_index[:, None] + offsets[None, :]
    ).clamp_(0, table.numel() - 1)
    candidates = table[candidate_index]

    totals = torch.tensor([1.0, 2.0, 4.0], device=blocks.device)
    divisor = (
        candidates[:, :, None, None, None, None]
        * totals[None, None, None, None, None, :]
    )
    expanded = blocks[:, None, :, :, :, None]
    mant = torch.round((expanded / divisor).abs() * 4.0).clamp_(0.0, 7.0) * 0.25
    reconstructed = torch.sign(expanded) * mant * divisor
    # [block, candidate, eight-group, four-group, total-scale]
    squared_error = (reconstructed - expanded).square()
    if importance is not None:
        squared_error = squared_error * importance[:, None, :, :, :, None]
    loss = squared_error.sum(dim=4)

    l2_one_cost = loss[..., 0:2].amin(dim=-1).sum(dim=-1)
    l2_two_cost = loss[..., 1:3].amin(dim=-1).sum(dim=-1)
    candidate_cost = torch.minimum(l2_one_cost, l2_two_cost).sum(dim=-1)
    best_candidate = candidate_cost.argmin(dim=-1)
    scale_factor = candidates.gather(1, best_candidate[:, None]).squeeze(1)

    chosen_divisor = (
        scale_factor[:, None, None, None, None]
        * totals[None, None, None, None, :]
    )
    expanded = blocks[:, :, :, :, None]
    mant = torch.round((expanded / chosen_divisor).abs() * 4.0).clamp_(0.0, 7.0) * 0.25
    reconstructed = torch.sign(expanded) * mant * chosen_divisor
    squared_error = (reconstructed - expanded).square()
    if importance is not None:
        squared_error = squared_error * importance[:, :, :, :, None]
    loss = squared_error.sum(dim=3)

    l2_one_cost = loss[..., 0:2].amin(dim=-1).sum(dim=-1)
    l2_two_cost = loss[..., 1:3].amin(dim=-1).sum(dim=-1)
    use_l2_two = l2_two_cost < l2_one_cost
    scale_lv2 = torch.where(use_l2_two, 2.0, 1.0).to(torch.float32)

    lv3_if_l2_one = torch.where(loss[..., 1] < loss[..., 0], 2.0, 1.0)
    lv3_if_l2_two = torch.where(loss[..., 2] < loss[..., 1], 2.0, 1.0)
    scale_lv3 = torch.where(
        use_l2_two[:, :, None],
        lv3_if_l2_two,
        lv3_if_l2_one,
    ).to(torch.float32)

    sign, mant = _encode_s1p2(
        blocks,
        scale_factor,
        scale_lv2,
        scale_lv3,
    )
    return scale_factor, scale_lv2, scale_lv3, sign, mant


def _quantize_hif4_search(
    x: torch.Tensor,
    radius: int = 2,
    chunk_blocks: int = 4096,
    importance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Demo quantizer: search nearby E6M2 scales and exact micro-scale choices."""
    original_shape = tuple(int(v) for v in x.shape)
    channels = int(original_shape[-1])
    if channels % 64 != 0:
        raise ValueError(f"last dimension {channels} is not divisible by 64")

    blocks = x.to(torch.float32).reshape(-1, 8, 2, 4)
    importance_blocks = None
    if importance is not None:
        importance = importance.to(device=blocks.device, dtype=torch.float32)
        importance = torch.broadcast_to(importance, original_shape)
        importance_blocks = importance.reshape(-1, 8, 2, 4)
    table = _e6m2_table(blocks.device)
    outputs: list[list[torch.Tensor]] = [[], [], [], [], []]
    for start in range(0, int(blocks.shape[0]), chunk_blocks):
        chunk_importance = None
        if importance_blocks is not None:
            chunk_importance = importance_blocks[start:start + chunk_blocks]
        result = _search_chunk(
            blocks[start:start + chunk_blocks],
            table,
            radius,
            chunk_importance,
        )
        for bucket, value in zip(outputs, result):
            bucket.append(value)

    scale_factor, scale_lv2, scale_lv3, sign, mant = [
        torch.cat(parts, dim=0) for parts in outputs
    ]
    return _pack_hif4(
        original_shape,
        scale_factor,
        scale_lv2,
        scale_lv3,
        sign,
        mant,
    )


def _attention_pair_scales(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance Q/K RMS with an exactly cancelling per-dimension transform."""
    q_per_kv = q_num_heads // kv_num_heads
    q_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float64)
    k_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float64)
    q_count = 0
    k_count = 0
    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample["q"]).to(torch.float32)
        k = dequantize_nvfp4(*sample["k"]).to(torch.float32)
        q = q.reshape(-1, kv_num_heads, q_per_kv, head_dim)
        k = k.reshape(-1, kv_num_heads, head_dim)
        q_energy += q.to(torch.float64).square().sum(dim=(0, 2)).cpu()
        k_energy += k.to(torch.float64).square().sum(dim=0).cpu()
        q_count += int(q.shape[0]) * q_per_kv
        k_count += int(k.shape[0])

    eps = torch.finfo(torch.float32).tiny
    q_rms = (q_energy / max(q_count, 1)).sqrt().to(torch.float32)
    k_rms = (k_energy / max(k_count, 1)).sqrt().to(torch.float32)
    pair_scale = torch.sqrt(
        q_rms.clamp_min(eps) / k_rms.clamp_min(eps)
    ).clamp(min=0.25, max=4.0)
    # A single conservative factor generalizes better than noisy per-dimension
    # calibration while still preserving QK^T exactly before quantization.
    global_scale = torch.exp(torch.log(pair_scale).mean() * 0.5).clamp(
        min=0.5,
        max=2.0,
    )
    # Tiny scale changes can move values across discrete S1P2 boundaries without
    # representing a real Q/K imbalance. Keep identity inside a 5% dead band.
    if abs(float(torch.log(global_scale).item())) < math.log(1.05):
        global_scale = torch.ones_like(global_scale)
    pair_scale = torch.ones_like(pair_scale) * global_scale
    q_scale = pair_scale.repeat_interleave(q_per_kv, dim=0).reshape(-1)
    k_scale = pair_scale.reshape(-1)
    return q_scale.contiguous(), k_scale.contiguous()


# =============================================================================
# 返回值公共说明
# =============================================================================
#
# HiF4Params
# -----------
# 所有需要返回 HiF4 量化结果的函数，都应返回一个 dict，并至少包含以下 5 个
# torch.Tensor：
#
#     {
#         "scale_factor": ...,
#         "scale_lv2":    ...,
#         "scale_lv3":    ...,
#         "sign":         ...,
#         "mant":         ...,
#     }
#
# 若原 Tensor shape 为 ``(*prefix, C)``，其中 C % 64 == 0，则五个字段的 shape 为：
#
#     scale_factor : (*prefix, C // 64, 1, 1, 1)
#     scale_lv2    : (*prefix, C // 64, 8, 1, 1)
#     scale_lv3    : (*prefix, C // 64, 8, 2, 1)
#     sign         : (*prefix, C // 64, 8, 2, 4)
#     mant         : (*prefix, C // 64, 8, 2, 4)
#
# 数值格式要求：
#
#     scale_factor : HiF4 E6M2 scale
#     scale_lv2    : 1 或 2
#     scale_lv3    : 1 或 2
#     sign         : -1、0 或 1
#     mant         : 0 ~ 1.75，步长 0.25
#
# 对应反量化关系为：
#
#     x_hat = sign * mant * scale_lv3 * scale_lv2 * scale_factor
#
# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
# calibration 函数返回的 activation_state / q_state / k_state / v_state 用于把
# calibration 阶段得到的信息传给对应的 dynamic quantization 函数。
#
# 推荐使用纯数据结构，例如：
#
#     None / bool / int / finite float / str
#     CPU torch.Tensor
#     list / tuple
#     dict[str, ...]
#
# 不要依赖自定义 Python 对象、可调用对象或外部可变状态。
#
# Linear 的 activation_state 可以包含固定 Weight 相关信息；例如选手可以根据自己的
# 算法保存 smooth scale、clip 参数、importance，或固定的 weight quantization 参数。
#


# =============================================================================
# 1. Linear calibration + Weight quantization
# =============================================================================

def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """使用 Weight 和 calibration Activation 完成离线校准，并量化 Weight。

    Args:
        weight_quant:
            Weight 的 NVFP4 value carrier。
            shape 通常为 ``[out_features, in_features]``。

        weight_scale:
            Weight 的 NVFP4 block scale。
            若 ``weight_quant.shape == [M, K]``，则通常为
            ``[M, K // 16]``。

        calib_activation_list:
            当前 Weight 对应的 calibration Activation 列表。

            每个元素均为一个二元 NVFP4 pair：

                (activation_quant, activation_scale)

            其中：

                activation_quant : [tokens, in_features]
                activation_scale : [tokens, in_features // 16]

            calibration 阶段可以同时利用 Weight 和这些 Activation 搜索：
            smooth scale、clip 参数、旋转参数、importance、量化参数等。

    Returns:
        必须返回：

            {
                "weight_params": HiF4Params,
                "activation_state": state,
            }

        weight_params:
            当前 Weight 的最终 HiF4 参数。
            它对应 ``weight_quant`` 解码后的原始 Weight shape。

        activation_state:
            传给 ``hif4_dynamic_quantize_activation`` 的 calibration state。

            这里可以保存后续在线 Activation 量化所需的固定信息，例如：
                - smooth scale
                - clip/search 参数
                - channel importance
                - rotation 参数
                - 固定 Weight 相关参数
                - 其他纯数据 calibration 结果

    Important:
        本函数必须自行实现 Weight 的 HiF4 量化算法。
        本模板不提供 HiF4 参考实现。
    """
    weight = dequantize_nvfp4(weight_quant, weight_scale).to(torch.float32)
    # R = D @ H4 is orthogonal.  Quantizing W @ R offline and A @ R online
    # preserves A @ W.T before quantization while spreading four-value outliers
    # within exactly one HiF4 level-3 group.
    rotation_signs = _deterministic_signs(int(weight.shape[-1]), seed=20260903)
    rotated_weight = _apply_hadamard_rotation(weight, rotation_signs, block_size=4)
    del calib_activation_list
    return {
        "weight_params": _quantize_hif4_search(rotated_weight, radius=2),
        "activation_state": {
            "rotation_block": 4,
            "rotation_signs": rotation_signs,
        },
    }


# =============================================================================
# 2. Dynamic Activation quantization
# =============================================================================

def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Activation 动态生成 HiF4 参数。

    Args:
        activation_quant:
            当前 Activation 的 NVFP4 value carrier，通常为
            ``[tokens, hidden_size]``。

        activation_scale:
            当前 Activation 的 NVFP4 block scale，通常为
            ``[tokens, hidden_size // 16]``。

        activation_state:
            与当前 Linear Weight 对应，由
            ``hif4_calibration_and_quantize_weight`` 返回的 state。

            dynamic quantization 可以使用这里保存的 calibration 信息和固定
            Weight 相关信息，再结合当前 Activation 自身进行搜索或动态决策。

    Returns:
        当前 Activation 对应的 HiF4Params。
        输出参数的逻辑 Tensor shape 必须与当前 Activation 一致。
    """
    activation = dequantize_nvfp4(
        activation_quant,
        activation_scale,
    ).to(torch.float32)
    if isinstance(activation_state, dict) and "rotation_signs" in activation_state:
        activation = _apply_hadamard_rotation(
            activation,
            activation_state["rotation_signs"],
            int(activation_state.get("rotation_block", 4)),
        )
    return _quantize_hif4_search(activation, radius=2)


# =============================================================================
# 3. Attention calibration
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """使用 calibration Q/K/V 生成 Q、K、V 后续动态量化所需的 state。

    Args:
        calib_qkv_list:
            calibration Q/K/V sample 列表。

            每个 sample 的标准结构为：

                {
                    "q": (q_quant, q_scale),
                    "k": (k_quant, k_scale),
                    "v": (v_quant, v_scale),
                }

            其中 quant Tensor 均为二维：

                q_quant : [seq_len, q_num_heads  * head_dim]
                k_quant : [seq_len, kv_num_heads * head_dim]
                v_quant : [seq_len, kv_num_heads * head_dim]

            对应 scale Tensor 的最后一维为 quant Tensor 最后一维 / 16。

        q_num_heads:
            Query head 数。

        kv_num_heads:
            Key / Value head 数。

        head_dim:
            每个 attention head 的维度。

    Returns:
        必须返回：

            {
                "q_state": q_state,
                "k_state": k_state,
                "v_state": v_state,
            }

        q_state:
            传给 ``hif4_dynamic_quantize_q``。

        k_state:
            传给 ``hif4_dynamic_quantize_k``。

        v_state:
            传给 ``hif4_dynamic_quantize_v``。

        三个 state 可以保存 calibration 阶段得到的固定纯数据参数，例如：
            - clip 参数
            - per-head / per-channel scale
            - rotation 参数
            - importance
            - 其他动态量化需要使用的 calibration 结果
    """
    if kv_num_heads <= 0 or q_num_heads <= 0 or q_num_heads % kv_num_heads != 0:
        raise ValueError(
            f"q_num_heads {q_num_heads} is not divisible by kv_num_heads {kv_num_heads}"
        )
    # H8 mixes the two four-value level-3 children under one level-2 scale.
    # Use the same rotation for each KV head and every Q head mapped to it, so
    # the pre-quantization Q @ K.T logits remain unchanged.
    rotation_block = next(
        (candidate for candidate in (8, 4, 2) if head_dim % candidate == 0),
        1,
    )
    k_signs = _deterministic_signs(kv_num_heads * head_dim, seed=424242).reshape(
        kv_num_heads, head_dim
    )
    q_per_kv = q_num_heads // kv_num_heads
    q_signs = k_signs.repeat_interleave(q_per_kv, dim=0).reshape(-1).contiguous()
    k_signs = k_signs.reshape(-1).contiguous()
    del calib_qkv_list
    return {
        "q_state": {
            "rotation_block": rotation_block,
            "rotation_signs": q_signs,
        },
        "k_state": {
            "rotation_block": rotation_block,
            "rotation_signs": k_signs,
        },
        "v_state": None,
    }


# =============================================================================
# 4. Dynamic Q quantization
# =============================================================================

def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Query Tensor 动态生成 HiF4 参数。

    Args:
        q_quant:
            Q 的 NVFP4 value carrier，shape 为
            ``[seq_len, q_num_heads * head_dim]``。

        q_scale:
            Q 的 NVFP4 block scale，shape 为
            ``[seq_len, q_num_heads * head_dim // 16]``。

        q_num_heads:
            Query head 数。

        head_dim:
            每个 Query head 的维度。

        q_state:
            ``hif4_calibration_attention`` 返回的 Q calibration state。

    Returns:
        当前 Q 对应的 HiF4Params。
    """
    del q_num_heads, head_dim
    q = dequantize_nvfp4(q_quant, q_scale).to(torch.float32)
    if isinstance(q_state, dict) and "pair_scale" in q_state:
        pair_scale = q_state["pair_scale"].to(
            device=q.device,
            dtype=torch.float32,
        )
        q = q / pair_scale
    if isinstance(q_state, dict) and "rotation_signs" in q_state:
        q = _apply_hadamard_rotation(
            q,
            q_state["rotation_signs"],
            int(q_state.get("rotation_block", 8)),
        )
    return _quantize_hif4_search(q, radius=2)


# =============================================================================
# 5. Dynamic K quantization
# =============================================================================

def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Key Tensor 动态生成 HiF4 参数。

    Args:
        k_quant:
            K 的 NVFP4 value carrier，shape 为
            ``[seq_len, kv_num_heads * head_dim]``。

        k_scale:
            K 的 NVFP4 block scale，shape 为
            ``[seq_len, kv_num_heads * head_dim // 16]``。

        kv_num_heads:
            Key / Value head 数。

        head_dim:
            每个 Key head 的维度。

        k_state:
            ``hif4_calibration_attention`` 返回的 K calibration state。

    Returns:
        当前 K 对应的 HiF4Params。
    """
    del kv_num_heads, head_dim
    k = dequantize_nvfp4(k_quant, k_scale).to(torch.float32)
    if isinstance(k_state, dict) and "pair_scale" in k_state:
        pair_scale = k_state["pair_scale"].to(
            device=k.device,
            dtype=torch.float32,
        )
        k = k * pair_scale
    if isinstance(k_state, dict) and "rotation_signs" in k_state:
        k = _apply_hadamard_rotation(
            k,
            k_state["rotation_signs"],
            int(k_state.get("rotation_block", 8)),
        )
    return _quantize_hif4_search(k, radius=2)


# =============================================================================
# 6. Dynamic V quantization
# =============================================================================

def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    """对当前 Value Tensor 动态生成 HiF4 参数。

    Args:
        v_quant:
            V 的 NVFP4 value carrier，shape 为
            ``[seq_len, kv_num_heads * head_dim]``。

        v_scale:
            V 的 NVFP4 block scale，shape 为
            ``[seq_len, kv_num_heads * head_dim // 16]``。

        kv_num_heads:
            Key / Value head 数。

        head_dim:
            每个 Value head 的维度。

        v_state:
            ``hif4_calibration_attention`` 返回的 V calibration state。

    Returns:
        当前 V 对应的 HiF4Params。
    """
    del kv_num_heads, head_dim, v_state
    v = dequantize_nvfp4(v_quant, v_scale).to(torch.float32)
    return _quantize_hif4_search(v, radius=2)
