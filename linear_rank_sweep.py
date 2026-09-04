from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from real_model_benchmark import (
    dequant_pair,
    from_hif4,
    load_solution,
    relative_score,
    standard_quantize,
)


def evaluate_linear(solution, case: dict) -> float:
    linear = case["linear"]
    weight = dequant_pair(solution, linear["weight"])
    calibration = solution.hif4_calibration_and_quantize_weight(
        *linear["weight"], linear["calib_activation_list"]
    )
    player_weight = from_hif4(calibration["weight_params"], tuple(weight.shape))
    standard_weight = from_hif4(standard_quantize(solution, weight), tuple(weight.shape))
    scores = []
    for pair in linear["test_activation_list"]:
        value = dequant_pair(solution, pair)
        reference = value @ weight.T
        standard_value = from_hif4(standard_quantize(solution, value), tuple(value.shape))
        player_value = from_hif4(
            solution.hif4_dynamic_quantize_activation(
                *pair, calibration["activation_state"]
            ),
            tuple(value.shape),
        )
        scores.append(relative_score(
            reference,
            standard_value @ standard_weight.T,
            player_value @ player_weight.T,
        ))
    return sum(scores) / len(scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="real_model_data/qwen2_5_0_5b.pt")
    parser.add_argument("--configs", nargs="+", default=["8x1", "16x1", "8x2"])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    bundle = torch.load(repo / args.dataset, map_location="cpu", weights_only=False)
    for config in args.configs:
        rank_text, sweeps_text = config.lower().split("x", 1)
        solution = load_solution(repo, "worktree")
        solution._HESSIAN_LOW_RANK = int(rank_text)
        solution._HESSIAN_LOW_RANK_SWEEPS = int(sweeps_text)
        started = time.perf_counter()
        scores = []
        for case in bundle["cases"]:
            score = evaluate_linear(solution, case)
            scores.append(score)
            print(f"{config:6s} {case['name']:26s} linear={score:+.5f}")
        elapsed = time.perf_counter() - started
        print(
            f"{config:6s} MEAN linear={sum(scores) / len(scores):+.5f} "
            f"elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
