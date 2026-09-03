from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from real_model_benchmark import (
    dequant_pair,
    evaluate_case,
    load_solution,
    to_nvfp4_pair,
)


def dummy_linear() -> dict:
    generator = torch.Generator().manual_seed(20260903)
    weight = to_nvfp4_pair(torch.randn(1, 64, generator=generator))
    samples = [
        to_nvfp4_pair(torch.randn(1, 64, generator=generator))
        for _ in range(10)
    ]
    return {
        "weight": weight,
        "calib_activation_list": samples[:5],
        "test_activation_list": samples[5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="real_model_data/qwen2_5_0_5b.pt")
    parser.add_argument("--public-datasets-dir")
    parser.add_argument("--use-calibration-as-test", action="store_true")
    parser.add_argument("--disable-attention-hessian", action="store_true")
    parser.add_argument("--show-stats", action="store_true")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=[0.25, 0.3125, 0.375, 0.4375, 0.5, 0.5625],
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    if args.public_datasets_dir:
        public_dir = Path(args.public_datasets_dir)
        attention = torch.load(
            public_dir / "attn.pt", map_location="cpu", weights_only=False
        )[0]
        cases = [{"name": "public", "linear": dummy_linear(), "attention": attention}]
    else:
        bundle = torch.load(repo / args.dataset, map_location="cpu", weights_only=False)
        cases = bundle["cases"]
    tiny_linear = dummy_linear()
    if args.show_stats:
        stats_solution = load_solution(repo, "worktree")
        for case in cases:
            values = {
                role: torch.cat([
                    dequant_pair(stats_solution, sample[role]).reshape(-1)
                    for sample in case["attention"]["calib"]
                ])
                for role in ("q", "k", "v")
            }
            fields = []
            for role, value in values.items():
                rms = value.square().mean().sqrt()
                fields.append(
                    f"{role}_rms={rms:.4g} {role}_peak/rms={value.abs().max()/rms:.3f}"
                )
            print(f"stats {case['name']}: " + " ".join(fields))
    for alpha in args.alphas:
        solution = load_solution(repo, "worktree")
        solution._ATTENTION_SMOOTH_ALPHA = alpha
        if args.disable_attention_hessian:
            solution._ATTENTION_HESSIAN_MIN_REPLACE_IMPROVEMENT = float("inf")
        results = []
        for original in cases:
            case = copy.copy(original)
            case["linear"] = tiny_linear
            if args.use_calibration_as_test:
                case["attention"] = copy.copy(original["attention"])
                case["attention"]["test"] = original["attention"]["calib"]
            result = evaluate_case(solution, case)
            results.append(result)
        full = sum(item["attention_full"] for item in results) / len(results)
        causal = sum(item["attention_causal"] for item in results) / len(results)
        layers = ", ".join(
            f"{item['attention_full']:+.4f}" for item in results
        )
        print(f"alpha={alpha:.4f} full={full:+.4f} causal={causal:+.4f} layers=[{layers}]")


if __name__ == "__main__":
    main()
