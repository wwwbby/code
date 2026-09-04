from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch

from attention_alpha_sweep import dummy_linear
from real_model_benchmark import evaluate_case, load_solution


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-datasets-dir")
    parser.add_argument(
        "--configs", nargs="+", default=["4x1", "4x2", "8x1", "8x2", "16x1"]
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.10])
    parser.add_argument("--token-caps", nargs="+", type=int, default=[64])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    if args.public_datasets_dir:
        attention = torch.load(
            Path(args.public_datasets_dir) / "attn.pt",
            map_location="cpu",
            weights_only=False,
        )[0]
        cases = [{"name": "public", "attention": attention}]
    else:
        bundle = torch.load(
            repo / "real_model_data/qwen2_5_0_5b.pt",
            map_location="cpu",
            weights_only=False,
        )
        cases = bundle["cases"]
    for config in args.configs:
        for threshold in args.thresholds:
            for token_cap in args.token_caps:
                rank_text, sweeps_text = config.split("x", 1)
                solution = load_solution(repo, "worktree")
                solution._ATTENTION_K_HESSIAN_RANK = int(rank_text)
                solution._ATTENTION_K_HESSIAN_SWEEPS = int(sweeps_text)
                solution._ATTENTION_K_HESSIAN_MIN_REPLACE_IMPROVEMENT = threshold
                solution._ATTENTION_K_HESSIAN_MAX_TOKENS = token_cap
                started = time.perf_counter()
                results = []
                for original in cases:
                    case = copy.copy(original)
                    case["linear"] = dummy_linear()
                    result = evaluate_case(solution, case)
                    results.append(result)
                full = sum(item["attention_full"] for item in results) / len(results)
                causal = sum(item["attention_causal"] for item in results) / len(results)
                print(
                    f"{config:5s} threshold={threshold:.3f} tokens={token_cap:3d} "
                    f"full={full:+.5f} causal={causal:+.5f} "
                    f"elapsed={time.perf_counter() - started:.2f}s"
                )


if __name__ == "__main__":
    main()
