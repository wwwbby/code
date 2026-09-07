from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from attention_alpha_sweep import dummy_linear
from real_model_benchmark import evaluate_case, load_solution, to_nvfp4_pair


@dataclass(frozen=True)
class Profile:
    name: str
    q_heads: int
    kv_heads: int
    head_dim: int
    length: int
    q_rms: float = 1.0
    k_rms: float = 1.0
    tail: float = 0.0
    channel_spread: float = 0.5


PROFILES = (
    Profile("gqa-d64-short", 8, 2, 64, 32, tail=0.01),
    Profile("mha-d64-long-tail", 4, 4, 64, 128, tail=0.025),
    Profile("gqa-d128-medium", 8, 2, 128, 64, channel_spread=0.8),
    Profile("mha-d128-q-heavy", 4, 4, 128, 96, q_rms=2.5),
    Profile("mqa-d256-k-heavy", 8, 1, 256, 128, k_rms=3.0),
    Profile("gqa-d256-long", 8, 2, 256, 256, tail=0.01, channel_spread=0.9),
)


def _channel_scale(
    generator: torch.Generator,
    heads: int,
    head_dim: int,
    spread: float,
) -> torch.Tensor:
    raw = torch.randn((heads, head_dim), generator=generator) * spread
    return raw.exp().clamp(0.2, 5.0)


def _sample(
    generator: torch.Generator,
    length: int,
    scale: torch.Tensor,
    rms: float,
    tail: float,
) -> torch.Tensor:
    value = torch.randn((length, *scale.shape), generator=generator)
    if tail > 0.0:
        mask = torch.rand(value.shape, generator=generator) < tail
        outlier = torch.randn(value.shape, generator=generator) * 6.0
        value = torch.where(mask, outlier, value)
    # A shared token factor creates head/channel correlation without baking in
    # any one model architecture.
    token = (0.75 + 0.5 * torch.rand((length, 1, 1), generator=generator))
    value = value * token * scale[None]
    value = value * (rms / value.square().mean().sqrt().clamp_min(1.0e-8))
    return value.reshape(length, -1)


def make_case(profile: Profile, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    q_scale = _channel_scale(
        generator, profile.q_heads, profile.head_dim, profile.channel_spread
    )
    k_scale = _channel_scale(
        generator, profile.kv_heads, profile.head_dim, profile.channel_spread
    )
    v_scale = _channel_scale(
        generator, profile.kv_heads, profile.head_dim, profile.channel_spread * 0.5
    )

    def qkv_sample() -> dict:
        return {
            "q": to_nvfp4_pair(_sample(
                generator,
                profile.length,
                q_scale,
                profile.q_rms,
                profile.tail,
            )),
            "k": to_nvfp4_pair(_sample(
                generator,
                profile.length,
                k_scale,
                profile.k_rms,
                profile.tail,
            )),
            "v": to_nvfp4_pair(_sample(
                generator,
                profile.length,
                v_scale,
                1.0,
                profile.tail,
            )),
        }

    samples = [qkv_sample() for _ in range(5)]
    return {
        "name": profile.name,
        "linear": dummy_linear(),
        "attention": {
            "q_num_heads": profile.q_heads,
            "kv_num_heads": profile.kv_heads,
            "head_dim": profile.head_dim,
            "calib": samples[:3],
            "test": samples[3:],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revisions", nargs="+", default=["worktree"])
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--min-improvement", type=float)
    parser.add_argument("--k-sweeps", type=int)
    parser.add_argument("--k-rank", type=int)
    parser.add_argument("--k-threshold", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--v-coupling", type=float)
    parser.add_argument("--v-updates", type=int)
    parser.add_argument("--k-strength", type=float)
    parser.add_argument("--k-center-mode", choices=("off", "always", "guarded"))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    cases = [make_case(profile, args.seed + index) for index, profile in enumerate(PROFILES)]
    for revision in args.revisions:
        solution = load_solution(repo, revision)
        if args.k_sweeps is not None:
            solution._ATTENTION_K_HESSIAN_SWEEPS = args.k_sweeps
        if args.k_rank is not None:
            solution._ATTENTION_K_HESSIAN_RANK = args.k_rank
        if args.k_threshold is not None:
            solution._ATTENTION_K_HESSIAN_MIN_REPLACE_IMPROVEMENT = args.k_threshold
        if args.alpha is not None:
            solution._ATTENTION_SMOOTH_ALPHA = args.alpha
            solution._ATTENTION_IMBALANCED_SMOOTH_ALPHA = args.alpha
        if args.v_coupling is not None:
            solution._V_TOKEN_COUPLING = args.v_coupling
        if args.v_updates is not None:
            solution._V_TOKEN_UPDATES = args.v_updates
        if args.k_strength is not None:
            solution._K_MANTISSA_STRENGTH = args.k_strength
        if args.k_center_mode is not None:
            solution._K_CENTER_MODE = args.k_center_mode
        if args.min_improvement is not None and hasattr(
            solution, "_ATTENTION_ALPHA_SELECTION_MIN_IMPROVEMENT"
        ):
            solution._ATTENTION_ALPHA_SELECTION_MIN_IMPROVEMENT = args.min_improvement
        rows = []
        for case in cases:
            result = evaluate_case(solution, copy.copy(case))
            rows.append(result)
            print(
                f"{revision:10s} {case['name']:21s} "
                f"full={result['attention_full']:+.5f} "
                f"causal={result['attention_causal']:+.5f}"
            )
        full = sum(row["attention_full"] for row in rows) / len(rows)
        causal = sum(row["attention_causal"] for row in rows) / len(rows)
        worst = min(min(row["attention_full"], row["attention_causal"]) for row in rows)
        print(
            f"{revision:10s} MEAN full={full:+.5f} causal={causal:+.5f} "
            f"worst={worst:+.5f}"
        )


if __name__ == "__main__":
    main()
