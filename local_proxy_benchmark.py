from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
import types

import torch


NVFP4_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def to_nvfp4_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(x.shape)
    blocks = x.to(torch.float32).reshape(*shape[:-1], shape[-1] // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / 6.0).clamp_min(2.0 ** -24)
    normalized = blocks / scale.unsqueeze(-1)
    distance = (normalized.unsqueeze(-1) - NVFP4_VALUES).abs()
    quant = NVFP4_VALUES[distance.argmin(dim=-1)]
    return quant.reshape(shape), scale


def from_hif4(params: dict[str, torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    value = (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    )
    return value.reshape(shape).to(torch.float32)


def dequant_pair(solution, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return solution.dequantize_nvfp4(*pair).to(torch.float32)


def standard_quantize(solution, value: torch.Tensor) -> dict[str, torch.Tensor]:
    """Call the module's plain HiF4 path across historical revisions."""
    for name in ("_quantize_hif4_direct", "_quantize_hif4_fast", "_quantize_hif4"):
        function = getattr(solution, name, None)
        if function is not None:
            return function(value)
    raise AttributeError("solution has no plain HiF4 quantizer")


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              q_heads: int, kv_heads: int, head_dim: int) -> torch.Tensor:
    seq = q.shape[0]
    qh = q.reshape(seq, q_heads, head_dim).transpose(0, 1)
    kh = k.reshape(seq, kv_heads, head_dim).transpose(0, 1)
    vh = v.reshape(seq, kv_heads, head_dim).transpose(0, 1)
    repeat = q_heads // kv_heads
    kh = kh.repeat_interleave(repeat, dim=0)
    vh = vh.repeat_interleave(repeat, dim=0)
    prob = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    return (prob @ vh).transpose(0, 1).reshape(seq, q_heads * head_dim)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).square().mean().item())


def score(std_mse: float, player_mse: float) -> float:
    if std_mse == 0.0:
        return 0.0
    return (std_mse - player_mse) / std_mse


def make_data(seed: int = 20260903):
    generator = torch.Generator().manual_seed(seed)

    def sample(shape, outlier_rate=0.015):
        x = torch.randn(shape, generator=generator)
        mask = torch.rand(shape, generator=generator) < outlier_rate
        outliers = torch.randn(shape, generator=generator) * 7.0
        return torch.where(mask, outliers, x)

    weight = to_nvfp4_pair(sample((128, 256), 0.02))
    linear_calib = [to_nvfp4_pair(sample((32, 256))) for _ in range(5)]
    linear_test = [to_nvfp4_pair(sample((32, 256))) for _ in range(5)]

    q_heads, kv_heads, head_dim, seq = 4, 2, 64, 32

    def qkv_sample():
        return {
            "q": to_nvfp4_pair(sample((seq, q_heads * head_dim))),
            "k": to_nvfp4_pair(sample((seq, kv_heads * head_dim))),
            "v": to_nvfp4_pair(sample((seq, kv_heads * head_dim))),
        }

    attn_calib = [qkv_sample() for _ in range(5)]
    attn_test = [qkv_sample() for _ in range(5)]
    return {
        "linear": {
            "weight": weight,
            "calib_activation_list": linear_calib,
            "test_activation_list": linear_test,
        },
        "attention": {
            "q_num_heads": q_heads,
            "kv_num_heads": kv_heads,
            "head_dim": head_dim,
            "calib": attn_calib,
            "test": attn_test,
        },
    }


def save_self_check_data(data, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save([data["linear"]], os.path.join(output_dir, "linear.pt"))
    torch.save([data["attention"]], os.path.join(output_dir, "attn.pt"))


def run(solution, data) -> None:
    linear = data["linear"]
    weight = dequant_pair(solution, linear["weight"])

    t0 = time.perf_counter()
    calibration = solution.hif4_calibration_and_quantize_weight(
        *linear["weight"], linear["calib_activation_list"]
    )
    player_weight = from_hif4(calibration["weight_params"], tuple(weight.shape))

    std_weight = from_hif4(standard_quantize(solution, weight), tuple(weight.shape))
    linear_scores = []
    for pair in linear["test_activation_list"]:
        activation = dequant_pair(solution, pair)
        reference = activation @ weight.T

        std_activation = from_hif4(
            standard_quantize(solution, activation), tuple(activation.shape)
        )
        std_output = std_activation @ std_weight.T

        player_params = solution.hif4_dynamic_quantize_activation(
            *pair, calibration["activation_state"]
        )
        player_activation = from_hif4(player_params, tuple(activation.shape))
        player_output = player_activation @ player_weight.T
        std_error = mse(std_output, reference)
        player_error = mse(player_output, reference)
        linear_scores.append((std_error, player_error, score(std_error, player_error)))
    linear_time = time.perf_counter() - t0

    attn = data["attention"]
    q_heads = attn["q_num_heads"]
    kv_heads = attn["kv_num_heads"]
    head_dim = attn["head_dim"]
    t0 = time.perf_counter()
    states = solution.hif4_calibration_attention(
        attn["calib"], q_heads, kv_heads, head_dim
    )
    attn_scores = []
    for sample in attn["test"]:
        q = dequant_pair(solution, sample["q"])
        k = dequant_pair(solution, sample["k"])
        v = dequant_pair(solution, sample["v"])
        reference = attention(q, k, v, q_heads, kv_heads, head_dim)

        std_q = from_hif4(standard_quantize(solution, q), tuple(q.shape))
        std_k = from_hif4(standard_quantize(solution, k), tuple(k.shape))
        std_v = from_hif4(standard_quantize(solution, v), tuple(v.shape))
        std_output = attention(std_q, std_k, std_v, q_heads, kv_heads, head_dim)

        player_q = from_hif4(
            solution.hif4_dynamic_quantize_q(
                *sample["q"], q_heads, head_dim, states["q_state"]
            ), tuple(q.shape)
        )
        player_k = from_hif4(
            solution.hif4_dynamic_quantize_k(
                *sample["k"], kv_heads, head_dim, states["k_state"]
            ), tuple(k.shape)
        )
        player_v = from_hif4(
            solution.hif4_dynamic_quantize_v(
                *sample["v"], kv_heads, head_dim, states["v_state"]
            ), tuple(v.shape)
        )
        player_output = attention(
            player_q, player_k, player_v, q_heads, kv_heads, head_dim
        )
        std_error = mse(std_output, reference)
        player_error = mse(player_output, reference)
        attn_scores.append((std_error, player_error, score(std_error, player_error)))
    attn_time = time.perf_counter() - t0

    def report(name, rows):
        std_mean = sum(row[0] for row in rows) / len(rows)
        player_mean = sum(row[1] for row in rows) / len(rows)
        score_sum = sum(row[2] for row in rows)
        score_mean = score_sum / len(rows)
        print(
            f"{name}: std_mse={std_mean:.8g}, player_mse={player_mean:.8g}, "
            f"score_sum={score_sum:.6f}, score_mean={score_mean:.6f}"
        )

    report("Linear", linear_scores)
    report("Attention", attn_scores)
    all_rows = linear_scores + attn_scores
    report("Combined", all_rows)
    print(f"Runtime: linear={linear_time:.3f}s, attention={attn_time:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution_dir", default=os.path.dirname(__file__))
    parser.add_argument("--save_datasets_dir")
    parser.add_argument("--datasets_dir")
    parser.add_argument("--revision")
    args = parser.parse_args()

    if args.revision:
        source = subprocess.check_output(
            ["git", "show", f"{args.revision}:solution.py"],
            cwd=os.path.abspath(args.solution_dir),
            text=True,
            encoding="utf-8",
        )
        solution = types.ModuleType(f"solution_{args.revision}")
        exec(compile(source, f"<{args.revision}:solution.py>", "exec"), solution.__dict__)
    else:
        sys.path.insert(0, os.path.abspath(args.solution_dir))
        import solution

    if args.datasets_dir:
        data = {
            "linear": torch.load(
                os.path.join(args.datasets_dir, "linear.pt"),
                map_location="cpu",
                weights_only=False,
            )[0],
            "attention": torch.load(
                os.path.join(args.datasets_dir, "attn.pt"),
                map_location="cpu",
                weights_only=False,
            )[0],
        }
    else:
        data = make_data()
    if args.save_datasets_dir:
        save_self_check_data(data, args.save_datasets_dir)
    run(solution, data)


if __name__ == "__main__":
    main()
