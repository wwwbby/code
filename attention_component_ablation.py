from __future__ import annotations

import copy
import time
from pathlib import Path

import torch

from attention_alpha_sweep import dummy_linear
from real_model_benchmark import evaluate_case, load_solution


def main() -> None:
    repo = Path(__file__).resolve().parent
    bundle = torch.load(
        repo / "real_model_data/qwen2_5_0_5b.pt",
        map_location="cpu",
        weights_only=False,
    )
    for variant in ("q", "k", "qk"):
        solution = load_solution(repo, "237b142")
        direct = load_solution(repo, "a649209")
        solution.hif4_dynamic_quantize_v = direct.hif4_dynamic_quantize_v
        if variant == "q":
            solution.hif4_dynamic_quantize_k = direct.hif4_dynamic_quantize_k
        elif variant == "k":
            solution.hif4_dynamic_quantize_q = direct.hif4_dynamic_quantize_q
        started = time.perf_counter()
        results = []
        for original in bundle["cases"]:
            case = copy.copy(original)
            case["linear"] = dummy_linear()
            result = evaluate_case(solution, case)
            results.append(result)
            print(
                f"{variant:2s} {case['name']:26s} "
                f"full={result['attention_full']:+.4f} "
                f"causal={result['attention_causal']:+.4f}"
            )
        full = sum(item["attention_full"] for item in results) / len(results)
        causal = sum(item["attention_causal"] for item in results) / len(results)
        print(
            f"{variant:2s} MEAN full={full:+.4f} causal={causal:+.4f} "
            f"elapsed={time.perf_counter() - started:.2f}s"
        )


if __name__ == "__main__":
    main()
