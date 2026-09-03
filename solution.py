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
    return _dequantize_nvfp4_fp32(quant_float, scale_float, blk_size).to(
        torch.bfloat16
    )


def _dequantize_nvfp4_fp32(
    quant_float: torch.Tensor,
    scale_float: torch.Tensor,
    blk_size: int = 16,
) -> torch.Tensor:
    """Decode the NVFP4 carrier without avoidable intermediate BF16 rounding."""
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )

    expected = tuple(quant_float.shape[:-1]) + (channels // blk_size,)
    if tuple(scale_float.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale_float.shape)} does not match {expected}"
        )
    x = quant_float.detach().to(torch.float32).unflatten(-1, (-1, blk_size))
    x = x * scale_float.detach().to(device=x.device, dtype=torch.float32).unsqueeze(-1)
    return x.flatten(-2, -1)


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
_GLOBAL_SCALE_MULTIPLIERS = (
    0.25,
    0.35,
    0.50,
    0.70,
    0.85,
    1.00,
    1.20,
    1.50,
    2.00,
    2.50,
    3.00,
    4.00,
)
_REFINEMENT_SCALE_MULTIPLIERS = (0.75, 0.875, 1.0, 1.125, 1.25)
_MIN_REFINEMENT_IMPROVEMENT = 0.01
_LINEAR_SMOOTH_ALPHA = 0.65
_ATTENTION_SMOOTH_ALPHA = 0.25
_SMOOTH_SCALE_MIN = 1.0 / 16.0
_SMOOTH_SCALE_MAX = 16.0
_V_IMPORTANCE_POWER = 0.25
_V_IMPORTANCE_BLEND = 0.25
_V_IMPORTANCE_MIN = 0.5
_V_IMPORTANCE_MAX = 2.0
_V_MIN_WEIGHTED_IMPROVEMENT = 0.01
_V_MAX_PLAIN_REGRESSION = 0.0025
_HESSIAN_DAMPING = 0.01
_HESSIAN_SWEEP_ROUNDS = 2
_HESSIAN_CHUNK_BLOCKS = 8192
_HESSIAN_MIN_IMPROVEMENT = 0.01


def _block_signs64(channels: int) -> torch.Tensor:
    """Build stable per-block Rademacher signs for a 64D Hadamard rotation."""
    if channels % 64 != 0:
        raise ValueError(f"channels {channels} is not divisible by 64")
    block = torch.arange(channels // 64, dtype=torch.int64)[:, None]
    lane = torch.arange(64, dtype=torch.int64)[None, :]
    hashed = (lane * 1103515245 + block * 12345 + 0x9E3779B9) & 0x7FFFFFFF
    return torch.where(((hashed >> 16) & 1) == 0, 1.0, -1.0).reshape(-1)


def _smooth_scale(
    left_abs_max: torch.Tensor,
    right_abs_max: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return a bounded reciprocal scale that balances two matmul operands."""
    left = left_abs_max.to(torch.float32).clamp_min(1.0e-6)
    right = right_abs_max.to(torch.float32).clamp_min(1.0e-6)
    scale = left.pow(alpha) / right.pow(1.0 - alpha)
    return torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).clamp(
        min=_SMOOTH_SCALE_MIN,
        max=_SMOOTH_SCALE_MAX,
    )


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


def _snap_to_e6m2(value: torch.Tensor) -> torch.Tensor:
    """Round positive FP32 values onto the checker's finite E6M2 grid."""
    value = torch.nan_to_num(
        value,
        nan=float(_E6M2_VALUES[0]),
        posinf=float(_E6M2_VALUES[-1]),
        neginf=float(_E6M2_VALUES[0]),
    ).clamp(min=float(_E6M2_VALUES[0]), max=float(_E6M2_VALUES[-1]))
    exponent = torch.floor(torch.log2(value))
    quantum = torch.pow(
        torch.tensor(2.0, dtype=value.dtype, device=value.device),
        exponent - 2.0,
    )
    return (torch.round(value / quantum) * quantum).clamp(
        min=float(_E6M2_VALUES[0]),
        max=float(_E6M2_VALUES[-1]),
    )


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


def _candidate_loss(
    blocks: torch.Tensor,
    candidates: torch.Tensor,
    importance: torch.Tensor | None,
) -> torch.Tensor:
    """Evaluate global scales after exactly optimizing the local scale tree."""
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
    return torch.minimum(l2_one_cost, l2_two_cost).sum(dim=-1)


def _search_chunk(
    blocks: torch.Tensor,
    importance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Minimize weighted SSE with broad and guarded refined E6M2 search."""
    peak = blocks.abs().amax(dim=(1, 2, 3))
    multipliers = torch.tensor(
        _GLOBAL_SCALE_MULTIPLIERS,
        dtype=torch.float32,
        device=blocks.device,
    )
    candidates = _snap_to_e6m2((peak / 7.0)[:, None] * multipliers[None, :])
    candidate_cost = _candidate_loss(blocks, candidates, importance)
    best_candidate = candidate_cost.argmin(dim=-1)
    row_index = torch.arange(int(blocks.shape[0]), device=blocks.device)
    baseline_scale = candidates[row_index, best_candidate]
    baseline_cost = candidate_cost[row_index, best_candidate]

    refinement = torch.tensor(
        _REFINEMENT_SCALE_MULTIPLIERS,
        dtype=torch.float32,
        device=blocks.device,
    )
    refined_candidates = _snap_to_e6m2(
        baseline_scale[:, None] * refinement[None, :]
    )
    refined_cost = _candidate_loss(blocks, refined_candidates, importance)
    best_refined = refined_cost.argmin(dim=-1)
    refined_scale = refined_candidates[row_index, best_refined]
    best_refined_cost = refined_cost[row_index, best_refined]
    use_refined = best_refined_cost < baseline_cost * (
        1.0 - _MIN_REFINEMENT_IMPROVEMENT
    )
    scale_factor = torch.where(use_refined, refined_scale, baseline_scale)

    totals = torch.tensor([1.0, 2.0, 4.0], device=blocks.device)
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
    chunk_blocks: int = 1024,
    importance: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Search legal global and local scales with an optional weighted loss."""
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
    outputs: list[list[torch.Tensor]] = [[], [], [], [], []]
    for start in range(0, int(blocks.shape[0]), chunk_blocks):
        chunk_importance = None
        if importance_blocks is not None:
            chunk_importance = importance_blocks[start:start + chunk_blocks]
        result = _search_chunk(
            blocks[start:start + chunk_blocks],
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


def _attention_weighted_v_energy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate each V channel's contribution under full and causal attention."""
    q_length = int(q.shape[0])
    key_length = int(k.shape[0])
    if q_length <= 256:
        positions = torch.arange(q_length, device=q.device)
    else:
        positions = torch.linspace(
            0,
            q_length - 1,
            steps=256,
            device=q.device,
        ).round().to(torch.int64)
    q_heads = q.reshape(q_length, q_num_heads, head_dim).permute(1, 0, 2)
    k_heads = k.reshape(key_length, kv_num_heads, head_dim).permute(1, 0, 2)
    v_energy = v.reshape(key_length, kv_num_heads, head_dim).permute(1, 0, 2).square()
    key_positions = torch.arange(key_length, device=q.device)
    q_per_kv = q_num_heads // kv_num_heads
    energy = torch.zeros((kv_num_heads, head_dim), device=q.device)
    usage_total = torch.zeros((kv_num_heads, 1), device=q.device)

    for kv_head in range(kv_num_heads):
        queries = q_heads[kv_head * q_per_kv:(kv_head + 1) * q_per_kv]
        key = k_heads[kv_head]
        for start in range(0, int(positions.numel()), 64):
            selected = positions[start:start + 64]
            logits = torch.matmul(
                queries[:, selected].float(),
                key.float().transpose(0, 1),
            ) / math.sqrt(float(head_dim))
            full_usage = torch.softmax(logits, dim=-1).square().sum(dim=(0, 1))
            if q_length == key_length:
                causal_limit = selected
            elif q_length == 1:
                causal_limit = torch.zeros_like(selected)
            else:
                causal_limit = torch.round(
                    selected.float() * float(key_length - 1) / float(q_length - 1)
                ).to(torch.int64)
            mask = key_positions[None, :] <= causal_limit[:, None]
            causal_usage = torch.softmax(
                logits.masked_fill(~mask[None], float("-inf")),
                dim=-1,
            ).square().sum(dim=(0, 1))
            usage = 0.5 * (full_usage + causal_usage)
            energy[kv_head] += (usage[:, None] * v_energy[kv_head]).sum(dim=0)
            usage_total[kv_head] += usage.sum()
    return energy, usage_total


def _finalize_v_importance(
    weighted_energy: torch.Tensor,
    usage_total: torch.Tensor,
) -> torch.Tensor:
    relative = weighted_energy / usage_total.clamp_min(1.0e-8)
    median = relative.median(dim=-1, keepdim=True).values
    compressed = (relative / (median + 1.0e-8)).clamp_min(0.0).pow(
        _V_IMPORTANCE_POWER
    ).clamp(min=_V_IMPORTANCE_MIN, max=_V_IMPORTANCE_MAX)
    importance = (1.0 - _V_IMPORTANCE_BLEND) + _V_IMPORTANCE_BLEND * compressed
    importance = torch.nan_to_num(importance, nan=1.0, posinf=1.0, neginf=1.0)
    blocks = importance.reshape(-1, 64)
    blocks = blocks / blocks.mean(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return blocks.reshape_as(importance).cpu().contiguous()


def _reconstruct_hif4(
    params: dict[str, torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    return (
        params["sign"]
        * params["mant"]
        * params["scale_lv2"]
        * params["scale_lv3"]
        * params["scale_factor"]
    ).reshape_as(reference)


def _local_scale_options(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enumerate the eight legal (level-2, left-level-3, right-level-3) choices."""
    lv2 = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        device=device,
    )
    lv3 = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        device=device,
    )
    return lv2, lv3, lv2[:, None] * lv3


def _evaluate_scale_candidates(
    abs_blocks: torch.Tensor,
    candidate_scales: torch.Tensor,
    importance: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return weighted errors and local-scale choices for global candidates."""
    _, _, local = _local_scale_options(abs_blocks.device)
    values = abs_blocks[:, None, :, None, :, :]
    divisors = (
        candidate_scales[:, :, None, None, None, None]
        * local[None, None, None, :, :, None]
    )
    mant = (torch.round(values / divisors * 4.0) * 0.25).clamp(0.0, 1.75)
    error = (values - mant * divisors).square()
    if importance is not None:
        error = error * importance[:, None, :, None, :, :]
    error = error.sum(dim=(-1, -2))
    local_error, local_choice = error.min(dim=-1)
    return local_error.sum(dim=-1), local_choice


def _build_block_hessian(
    transformed_activations: list[torch.Tensor],
    channels: int,
) -> torch.Tensor:
    """Build a normalized, damped 64D covariance for each input block."""
    rows = torch.cat(
        [value.reshape(-1, channels) for value in transformed_activations],
        dim=0,
    )
    block_count = channels // 64
    stacked = rows.reshape(-1, block_count, 64)
    hessian = torch.einsum("nbx,nby->bxy", stacked, stacked) / max(
        int(rows.shape[0]),
        1,
    )
    trace = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(dim=-1)
    normalization = (trace / 64.0).clamp_min(1.0e-8)
    eye = torch.eye(64, dtype=torch.float32, device=rows.device)[None]
    return hessian / normalization[:, None, None] + _HESSIAN_DAMPING * eye


def _materialize_block_choice(
    values: torch.Tensor,
    global_scale: torch.Tensor,
    local_choice: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Decode one selected global/local scale assignment for 64D blocks."""
    lv2, lv3, local = _local_scale_options(values.device)
    total = global_scale[:, None, None, None] * local[local_choice][:, :, :, None]
    blocks = values.reshape(-1, 8, 2, 4)
    mant = (torch.round(blocks.abs() / total * 4.0) * 0.25).clamp(0.0, 1.75)
    sign = torch.where(mant == 0.0, 0.0, torch.sign(blocks))
    reconstructed = (sign * mant * total).reshape(-1, 64)
    params = {
        "scale_factor": global_scale[:, None, None, None],
        "scale_lv2": lv2[local_choice][:, :, None, None],
        "scale_lv3": lv3[local_choice][:, :, :, None],
        "sign": sign,
        "mant": mant,
    }
    return reconstructed, params


def _group_reconstruction_cache(
    values: torch.Tensor,
    global_scale: torch.Tensor,
) -> list[torch.Tensor]:
    """Precompute all eight local reconstructions for every 8-value group."""
    _, _, local = _local_scale_options(values.device)
    abs_values = values.abs()
    signs = torch.sign(values)
    caches: list[torch.Tensor] = []
    for group in range(8):
        piece = abs_values[:, group * 8:(group + 1) * 8].reshape(-1, 2, 4)
        divisor = global_scale[:, None, None, None] * local[None, :, :, None]
        mant = (torch.round(piece[:, None] / divisor * 4.0) * 0.25).clamp(
            0.0,
            1.75,
        )
        group_sign = signs[:, group * 8:(group + 1) * 8].reshape(-1, 1, 2, 4)
        group_sign = torch.where(mant == 0.0, 0.0, group_sign)
        caches.append((group_sign * mant * divisor).reshape(-1, 8, 8))
    return caches


def _hessian_loss(
    error: torch.Tensor,
    hessian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hessian_error = torch.bmm(error[:, None, :], hessian).squeeze(1)
    return (hessian_error * error).sum(dim=-1), hessian_error


def _sweep_hessian_local_scales(
    values: torch.Tensor,
    global_scale: torch.Tensor,
    initial_choice: torch.Tensor,
    hessian: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run deterministic coordinate descent over the eight local scale groups."""
    choices = initial_choice.clone()
    reconstructed, _ = _materialize_block_choice(values, global_scale, choices)
    error = values - reconstructed
    loss, hessian_error = _hessian_loss(error, hessian)
    changes = torch.zeros(int(values.shape[0]), dtype=torch.int64, device=values.device)
    caches = _group_reconstruction_cache(values, global_scale)

    for _ in range(_HESSIAN_SWEEP_ROUNDS):
        for group in range(8):
            start = group * 8
            old_reconstruction = reconstructed[:, start:start + 8]
            old_error = error[:, start:start + 8]
            delta = caches[group] - old_reconstruction[:, None, :]
            diagonal_hessian = hessian[:, start:start + 8, start:start + 8]
            quadratic = (delta @ diagonal_hessian.transpose(-1, -2) * delta).sum(
                dim=-1
            )
            linear = 2.0 * (
                delta * hessian_error[:, None, start:start + 8]
            ).sum(dim=-1)
            delta_loss = quadratic - linear
            best_choice = delta_loss.argmin(dim=-1)
            best_delta_loss = delta_loss.gather(1, best_choice[:, None]).squeeze(1)
            chosen = torch.where(
                best_delta_loss < 0.0,
                best_choice,
                choices[:, group],
            )
            changed = chosen != choices[:, group]
            changes += changed.to(torch.int64)
            chosen_delta = delta.gather(
                1,
                chosen[:, None, None].expand(-1, 1, 8),
            ).squeeze(1)
            if bool(changed.any()):
                reconstructed[changed, start:start + 8] = caches[group][
                    changed,
                    chosen[changed],
                ]
                error[changed, start:start + 8] = (
                    old_error[changed] - chosen_delta[changed]
                )
                hessian_error -= torch.bmm(
                    hessian[:, :, start:start + 8],
                    chosen_delta[:, :, None],
                ).squeeze(-1)
                loss = loss + torch.where(
                    changed,
                    best_delta_loss,
                    torch.zeros((), device=values.device),
                )
            choices[:, group] = chosen

    reconstructed, params = _materialize_block_choice(values, global_scale, choices)
    loss, _ = _hessian_loss(values - reconstructed, hessian)
    return loss, changes, params


def _select_hessian_winner(
    current: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]] | None,
    loss: torch.Tensor,
    changes: torch.Tensor,
    scale: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Select deterministically by loss, edit count, then smaller global scale."""
    if current is None:
        return loss, changes, scale, params
    old_loss, old_changes, old_scale, old_params = current
    better = (
        (loss < old_loss)
        | ((loss == old_loss) & (changes < old_changes))
        | ((loss == old_loss) & (changes == old_changes) & (scale < old_scale))
    )
    condition = better[:, None, None, None]
    selected_params = {
        key: torch.where(condition, params[key], old_params[key])
        for key in old_params
    }
    return (
        torch.where(better, loss, old_loss),
        torch.where(better, changes, old_changes),
        torch.where(better, scale, old_scale),
        selected_params,
    )


def _quantize_weight_with_block_hessian(
    weight: torch.Tensor,
    activation_second: torch.Tensor,
    hessian_by_input_block: torch.Tensor,
    baseline: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Improve Linear weight blocks under their calibration-output Hessian."""
    rows, channels = weight.shape
    block_count = channels // 64
    values = weight.reshape(rows, block_count, 64).reshape(-1, 64)
    block_index = torch.arange(int(values.shape[0]), device=weight.device) % block_count
    importance = activation_second.reshape(block_count, 64)
    flat_baseline = {
        key: tensor.reshape(rows * block_count, *tensor.shape[2:])
        for key, tensor in baseline.items()
    }
    baseline_values = (
        flat_baseline["sign"]
        * flat_baseline["mant"]
        * flat_baseline["scale_lv2"]
        * flat_baseline["scale_lv3"]
        * flat_baseline["scale_factor"]
    ).reshape(-1, 64)
    output = {key: torch.empty_like(value) for key, value in flat_baseline.items()}
    coarse = torch.tensor(_GLOBAL_SCALE_MULTIPLIERS, device=weight.device)
    refine = torch.tensor(_REFINEMENT_SCALE_MULTIPLIERS, device=weight.device)

    for start in range(0, int(values.shape[0]), _HESSIAN_CHUNK_BLOCKS):
        end = min(start + _HESSIAN_CHUNK_BLOCKS, int(values.shape[0]))
        batch = values[start:end]
        batch_hessian = hessian_by_input_block[block_index[start:end]]
        baseline_loss, _ = _hessian_loss(
            batch - baseline_values[start:end],
            batch_hessian,
        )
        batch_importance = importance[block_index[start:end]].reshape(-1, 8, 2, 4)
        peak_scale = batch.abs().amax(dim=-1) / 7.0
        candidate_scales = _snap_to_e6m2(peak_scale[:, None] * coarse[None])
        _, initial_choices = _evaluate_scale_candidates(
            batch.abs().reshape(-1, 8, 2, 4),
            candidate_scales,
            batch_importance,
        )
        winner = None
        for candidate in range(len(_GLOBAL_SCALE_MULTIPLIERS)):
            scale = candidate_scales[:, candidate]
            result_loss, changes, params = _sweep_hessian_local_scales(
                batch,
                scale,
                initial_choices[:, candidate],
                batch_hessian,
            )
            winner = _select_hessian_winner(
                winner,
                result_loss,
                changes,
                scale,
                params,
            )

        assert winner is not None
        winner_scale = winner[2]
        refined_scales = _snap_to_e6m2(winner_scale[:, None] * refine[None])
        _, refined_choices = _evaluate_scale_candidates(
            batch.abs().reshape(-1, 8, 2, 4),
            refined_scales,
            batch_importance,
        )
        refined_winner = None
        for candidate in range(len(_REFINEMENT_SCALE_MULTIPLIERS)):
            scale = refined_scales[:, candidate]
            result_loss, changes, params = _sweep_hessian_local_scales(
                batch,
                scale,
                refined_choices[:, candidate],
                batch_hessian,
            )
            refined_winner = _select_hessian_winner(
                refined_winner,
                result_loss,
                changes,
                scale,
                params,
            )

        assert refined_winner is not None
        improved_loss, _, _, improved_params = refined_winner
        use_improved = improved_loss < baseline_loss * (1.0 - _HESSIAN_MIN_IMPROVEMENT)
        condition = use_improved[:, None, None, None]
        for key in output:
            output[key][start:end] = torch.where(
                condition,
                improved_params[key],
                flat_baseline[key][start:end],
            )

    return {
        key: value.reshape(rows, block_count, *value.shape[1:]).contiguous()
        for key, value in output.items()
    }


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
    weight = _dequantize_nvfp4_fp32(weight_quant, weight_scale)
    channels = int(weight.shape[-1])
    if not calib_activation_list:
        raise ValueError("calib_activation_list must not be empty")

    activation_abs_max = torch.zeros(
        channels,
        dtype=torch.float32,
        device=weight.device,
    )
    decoded_activations: list[torch.Tensor] = []
    for quant, scale in calib_activation_list:
        activation = _dequantize_nvfp4_fp32(quant, scale)
        decoded_activations.append(activation)
        activation_abs_max = torch.maximum(
            activation_abs_max,
            activation.abs().reshape(-1, channels).amax(dim=0),
        )

    smooth = _smooth_scale(
        activation_abs_max,
        weight.abs().amax(dim=0),
        _LINEAR_SMOOTH_ALPHA,
    )
    rotation_signs = _block_signs64(channels)
    transformed_weight = _apply_hadamard_rotation(
        weight * smooth,
        rotation_signs,
        block_size=64,
    )

    activation_second = torch.zeros(
        channels,
        dtype=torch.float32,
        device=weight.device,
    )
    activation_count = 0
    transformed_activations: list[torch.Tensor] = []
    for activation in decoded_activations:
        transformed = _apply_hadamard_rotation(
            activation / smooth,
            rotation_signs,
            block_size=64,
        )
        transformed_activations.append(transformed)
        activation_second += transformed.reshape(-1, channels).square().sum(dim=0)
        activation_count += transformed.numel() // channels
    activation_second = (activation_second / max(activation_count, 1)).clamp_min(
        1.0e-8
    )
    activation_importance = transformed_weight.square().sum(dim=0).clamp_min(1.0e-8)
    baseline_weight_params = _quantize_hif4_search(
        transformed_weight,
        importance=activation_second,
    )
    weight_params = _quantize_weight_with_block_hessian(
        transformed_weight,
        activation_second,
        _build_block_hessian(transformed_activations, channels),
        baseline_weight_params,
    )
    return {
        "weight_params": weight_params,
        "activation_state": {
            "rotation_block": 64,
            "rotation_signs": rotation_signs,
            "smooth_scale": smooth.detach().cpu().contiguous(),
            "importance": activation_importance.detach().cpu().contiguous(),
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
    activation = _dequantize_nvfp4_fp32(
        activation_quant,
        activation_scale,
    )
    if isinstance(activation_state, dict) and "smooth_scale" in activation_state:
        smooth = activation_state["smooth_scale"].to(
            device=activation.device,
            dtype=torch.float32,
        )
        activation = activation / smooth
    if isinstance(activation_state, dict) and "rotation_signs" in activation_state:
        activation = _apply_hadamard_rotation(
            activation,
            activation_state["rotation_signs"],
            int(activation_state.get("rotation_block", 4)),
        )
    importance = None
    if isinstance(activation_state, dict) and "importance" in activation_state:
        importance = activation_state["importance"]
    return _quantize_hif4_search(
        activation,
        importance=importance,
    )


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
    q_per_kv = q_num_heads // kv_num_heads
    q_abs_max = torch.zeros((q_num_heads, head_dim), dtype=torch.float32)
    k_abs_max = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    v_weighted_energy = torch.zeros((kv_num_heads, head_dim), dtype=torch.float32)
    v_usage_total = torch.zeros((kv_num_heads, 1), dtype=torch.float32)
    for sample in calib_qkv_list:
        q = _dequantize_nvfp4_fp32(*sample["q"])
        k = _dequantize_nvfp4_fp32(*sample["k"])
        v = _dequantize_nvfp4_fp32(*sample["v"])
        q_abs_max = torch.maximum(
            q_abs_max,
            q.reshape(-1, q_num_heads, head_dim).abs().amax(dim=0).cpu(),
        )
        k_abs_max = torch.maximum(
            k_abs_max,
            k.reshape(-1, kv_num_heads, head_dim).abs().amax(dim=0).cpu(),
        )
        sample_energy, sample_usage = _attention_weighted_v_energy(
            q,
            k,
            v,
            q_num_heads,
            kv_num_heads,
            head_dim,
        )
        v_weighted_energy += sample_energy.cpu()
        v_usage_total += sample_usage.cpu()

    q_group_max = q_abs_max.reshape(
        kv_num_heads,
        q_per_kv,
        head_dim,
    ).amax(dim=1)
    kv_smooth = _smooth_scale(
        q_group_max,
        k_abs_max,
        _ATTENTION_SMOOTH_ALPHA,
    ).cpu()
    q_smooth = kv_smooth[:, None, :].expand(
        kv_num_heads,
        q_per_kv,
        head_dim,
    ).reshape(q_num_heads, head_dim).contiguous()

    # Reset the signed H64 pattern in every head and share it across mapped
    # Q/K heads.  Q/s and K*s followed by the same orthogonal transform leave
    # the full-precision attention logits unchanged before quantization.
    head_signs = _block_signs64(head_dim).reshape(1, head_dim)
    q_signs = head_signs.expand(q_num_heads, head_dim).reshape(-1).contiguous()
    k_signs = head_signs.expand(kv_num_heads, head_dim).reshape(-1).contiguous()
    return {
        "q_state": {
            "rotation_block": 64,
            "rotation_signs": q_signs,
            "smooth_scale": q_smooth,
        },
        "k_state": {
            "rotation_block": 64,
            "rotation_signs": k_signs,
            "smooth_scale": kv_smooth.contiguous(),
        },
        "v_state": {
            "importance": _finalize_v_importance(
                v_weighted_energy,
                v_usage_total,
            ),
        },
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
    q = _dequantize_nvfp4_fp32(q_quant, q_scale)
    if isinstance(q_state, dict) and "smooth_scale" in q_state:
        smooth = q_state["smooth_scale"].to(
            device=q.device,
            dtype=torch.float32,
        )
        q = (q.reshape(-1, q_num_heads, head_dim) / smooth).reshape(q.shape)
    if isinstance(q_state, dict) and "rotation_signs" in q_state:
        q = _apply_hadamard_rotation(
            q,
            q_state["rotation_signs"],
            int(q_state.get("rotation_block", 8)),
        )
    return _quantize_hif4_search(q)


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
    k = _dequantize_nvfp4_fp32(k_quant, k_scale)
    if isinstance(k_state, dict) and "smooth_scale" in k_state:
        smooth = k_state["smooth_scale"].to(
            device=k.device,
            dtype=torch.float32,
        )
        k = (k.reshape(-1, kv_num_heads, head_dim) * smooth).reshape(k.shape)
    if isinstance(k_state, dict) and "rotation_signs" in k_state:
        k = _apply_hadamard_rotation(
            k,
            k_state["rotation_signs"],
            int(k_state.get("rotation_block", 8)),
        )
    return _quantize_hif4_search(k)


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
    v = _dequantize_nvfp4_fp32(v_quant, v_scale)
    if not isinstance(v_state, dict) or "importance" not in v_state:
        return _quantize_hif4_search(v)

    importance = v_state["importance"].to(
        device=v.device,
        dtype=torch.float32,
    ).reshape(-1)
    if importance.numel() != kv_num_heads * head_dim:
        raise ValueError("V importance shape does not match attention dimensions")
    baseline = _quantize_hif4_search(v)
    candidate = _quantize_hif4_search(v, importance=importance)
    baseline_error = (v - _reconstruct_hif4(baseline, v)).square().reshape(-1, 64)
    candidate_error = (v - _reconstruct_hif4(candidate, v)).square().reshape(-1, 64)
    weight_blocks = importance.reshape(-1, 64).unsqueeze(0).expand(
        int(v.shape[0]),
        -1,
        -1,
    ).reshape(-1, 64)
    baseline_plain = baseline_error.sum(dim=-1)
    candidate_plain = candidate_error.sum(dim=-1)
    baseline_weighted = (baseline_error * weight_blocks).sum(dim=-1)
    candidate_weighted = (candidate_error * weight_blocks).sum(dim=-1)
    use_candidate = (
        candidate_weighted
        < baseline_weighted * (1.0 - _V_MIN_WEIGHTED_IMPROVEMENT)
    ) & (
        candidate_plain
        <= baseline_plain * (1.0 + _V_MAX_PLAIN_REGRESSION)
    )
    mask = use_candidate.reshape(
        int(v.shape[0]),
        int(v.shape[-1]) // 64,
        1,
        1,
        1,
    )
    return {
        key: torch.where(mask, candidate[key], baseline[key]).contiguous()
        for key in baseline
    }
