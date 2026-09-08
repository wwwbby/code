"""Calibrate the random trend dataset against observed contest revisions.

The hidden set is not published and may itself be randomly generated.  The
primary score here therefore uses only fixed-seed random, model-shaped cases.
The supplied sparse sample and captured Qwen bundle are optional guard sets;
they are never allowed to determine the fitted trend weights.

Typical use from this directory::

    python online_trend_dataset.py --output online_trend_dataset.json
    python online_trend_dataset.py --evaluate --output online_trend_dataset.json \
        --revisions d75e03a 5b922c8 a649209 def4524 3c40705 988385e \
        fe4b879 768a670 8d88fbc 3e93bd2

The first command records the deterministic case recipe.  The second command
evaluates the selected historical revisions, fits a balanced source mixture,
and stores the calibration summary.  ``--score worktree`` can then score a
new candidate without changing the manifest recipe.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from real_model_benchmark import (
    attention,
    dequant_pair,
    from_hif4,
    load_solution,
    relative_score,
    standard_quantize,
)
from robust_attention_benchmark import PROFILES, make_case as make_attention_case
from robust_linear_benchmark import make_cases as make_linear_cases
from random_exam_dataset import (
    ATTENTION_PROFILES as EXAM_ATTENTION_PROFILES,
    LINEAR_PROFILES as EXAM_LINEAR_PROFILES,
    build as build_exam_cases,
)


ONLINE_RESULTS = (
    ("d75e03a", 15300, 261, "measured"),
    ("5b922c8", 15600, 248, "measured"),
    ("a649209", 16000, 243, "measured"),
    ("237b142", None, 300, "timeout"),
    ("b7248c6", 16000, None, "measured"),
    ("def4524", 16775, 242, "measured"),
    ("097c2e5", 16775, 242, "measured"),
    ("e0a19b0", 16700, 250, "measured"),
    ("3c40705", 16991, 250, "measured"),
    ("988385e", 17440, 252, "measured"),
    ("be6ffae", 17508, 254, "measured"),
    ("fe4b879", 17675, None, "measured"),
    ("768a670", 18178, 233, "measured"),
    ("8d88fbc", 18182, 230, "measured"),
    ("3e93bd2", 18417, 232, "measured"),
)

# Each exam-like profile receives an independent derived seed.  Extra legacy
# robust seeds are opt-in; the default sweep stays fast and matches the public
# variable-length layout more closely.
DEFAULT_ATTENTION_SEEDS: tuple[int, ...] = ()
DEFAULT_LINEAR_SEEDS: tuple[int, ...] = ()
SOURCE_ORDER = (
    "attn_random",
    "linear_random",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _public_case(repo: Path) -> dict[str, Any]:
    attention_case = torch.load(
        repo.parent / "organizer_sparse" / "data" / "attn.pt",
        map_location="cpu",
        weights_only=False,
    )[0]
    linear_case = torch.load(
        repo.parent / "organizer_sparse" / "data" / "linear.pt",
        map_location="cpu",
        weights_only=False,
    )[0]
    return {
        "name": "public-sparse",
        "linear": linear_case,
        "attention": attention_case,
    }


def _qwen_cases(repo: Path) -> list[dict[str, Any]]:
    bundle = torch.load(
        repo / "real_model_data" / "qwen2_5_0_5b.pt",
        map_location="cpu",
        weights_only=False,
    )
    return list(bundle["cases"])


def materialize_cases(
    repo: Path | None = None,
    attention_seeds: tuple[int, ...] = DEFAULT_ATTENTION_SEEDS,
    linear_seeds: tuple[int, ...] = DEFAULT_LINEAR_SEEDS,
    include_guards: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic case groups without duplicating source tensors."""
    repo = _repo_root() if repo is None else Path(repo)
    exam_linear, exam_attention = build_exam_cases(20260908)
    attention_cases: list[dict[str, Any]] = []
    for seed in attention_seeds:
        for index, profile in enumerate(PROFILES):
            case = make_attention_case(profile, seed + index)
            case["name"] = f"robust-attn-s{seed}-{profile.name}"
            attention_cases.append(case["attention"])
    linear_cases: list[dict[str, Any]] = []
    for seed in linear_seeds:
        for case in make_linear_cases(seed):
            item = dict(case)
            item["name"] = f"robust-linear-s{seed}-{case['name']}"
            linear_cases.append(item)
    groups = {
        "attn_random": exam_attention + attention_cases,
        "linear_random": exam_linear + linear_cases,
    }
    if include_guards:
        public = _public_case(repo)
        qwen = _qwen_cases(repo)
        groups.update({
            "attn_public": [public["attention"]],
            "attn_qwen": [case["attention"] for case in qwen],
            "linear_public": [public["linear"]],
            "linear_qwen": [case["linear"] for case in qwen],
        })
    return groups


def _score_linear(solution: Any, case: dict[str, Any]) -> float:
    calibration_pairs = case.get(
        "calib_activation_list", case.get("calibration", ())
    )
    test_pairs = case.get("test_activation_list", case.get("test", ()))
    weight = dequant_pair(solution, case["weight"])
    calibration = solution.hif4_calibration_and_quantize_weight(
        *case["weight"], calibration_pairs
    )
    player_weight = from_hif4(
        calibration["weight_params"], tuple(weight.shape)
    )
    standard_weight = from_hif4(
        standard_quantize(solution, weight), tuple(weight.shape)
    )
    scores = []
    for pair in test_pairs:
        value = dequant_pair(solution, pair)
        reference = value @ weight.T
        standard_value = from_hif4(
            standard_quantize(solution, value), tuple(value.shape)
        )
        player_value = from_hif4(
            solution.hif4_dynamic_quantize_activation(
                *pair, calibration["activation_state"]
            ),
            tuple(value.shape),
        )
        scores.append(
            relative_score(
                reference,
                standard_value @ standard_weight.T,
                player_value @ player_weight.T,
            )
        )
    return sum(scores) / max(len(scores), 1)


def _score_attention(solution: Any, case: dict[str, Any]) -> tuple[float, float]:
    q_heads = case["q_num_heads"]
    kv_heads = case["kv_num_heads"]
    head_dim = case["head_dim"]
    states = solution.hif4_calibration_attention(
        case["calib"], q_heads, kv_heads, head_dim
    )
    full_scores: list[float] = []
    causal_scores: list[float] = []
    for sample in case["test"]:
        original = {
            name: dequant_pair(solution, sample[name])
            for name in ("q", "k", "v")
        }
        standard = {
            name: from_hif4(
                standard_quantize(solution, original[name]),
                tuple(original[name].shape),
            )
            for name in ("q", "k", "v")
        }
        functions = {
            "q": solution.hif4_dynamic_quantize_q,
            "k": solution.hif4_dynamic_quantize_k,
            "v": solution.hif4_dynamic_quantize_v,
        }
        heads = {"q": q_heads, "k": kv_heads, "v": kv_heads}
        player = {
            name: from_hif4(
                functions[name](
                    *sample[name], heads[name], head_dim,
                    states[f"{name}_state"],
                ),
                tuple(original[name].shape),
            )
            for name in ("q", "k", "v")
        }
        reference = {
            causal: attention(
                **original,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                causal=causal,
            )
            for causal in (False, True)
        }
        baseline = {
            causal: attention(
                **standard,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                causal=causal,
            )
            for causal in (False, True)
        }
        candidate = {
            causal: attention(
                **player,
                q_heads=q_heads,
                kv_heads=kv_heads,
                head_dim=head_dim,
                causal=causal,
            )
            for causal in (False, True)
        }
        full_scores.append(relative_score(reference[False], baseline[False], candidate[False]))
        causal_scores.append(relative_score(reference[True], baseline[True], candidate[True]))
    return (
        sum(full_scores) / max(len(full_scores), 1),
        sum(causal_scores) / max(len(causal_scores), 1),
    )


def evaluate_revision(
    solution: Any,
    cases: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for source in tuple(name for name in cases if name.startswith("linear_")):
        rows = [_score_linear(solution, case) for case in cases[source]]
        values[source] = sum(rows) / max(len(rows), 1)
    for source in tuple(name for name in cases if name.startswith("attn_")):
        rows = [_score_attention(solution, case) for case in cases[source]]
        values[f"{source}_full"] = sum(row[0] for row in rows) / max(len(rows), 1)
        values[f"{source}_causal"] = sum(row[1] for row in rows) / max(len(rows), 1)
        values[source] = 0.5 * (values[f"{source}_full"] + values[f"{source}_causal"])
    return values


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = 0.5 * (index + end - 1)
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def _rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    a = torch.tensor(_rank(left), dtype=torch.float64)
    b = torch.tensor(_rank(right), dtype=torch.float64)
    a -= a.mean()
    b -= b.mean()
    denominator = (a.square().sum() * b.square().sum()).sqrt().item()
    return float((a * b).sum().item() / denominator) if denominator else 0.0


def _pairwise_accuracy(predicted: list[float], target: list[float]) -> float:
    correct = 0
    total = 0
    for i in range(len(target)):
        for j in range(i + 1, len(target)):
            if target[i] == target[j]:
                continue
            total += 1
            correct += int((predicted[i] - predicted[j]) * (target[i] - target[j]) > 0)
    return correct / max(total, 1)


def fit_source_weights(
    metrics: dict[str, dict[str, float]],
    revisions: list[str],
    online_scores: dict[str, float],
    seed: int = 20260908,
) -> dict[str, Any]:
    """Fit the Attention/Linear mixture over random source groups only."""
    rows = [revision for revision in revisions if revision in online_scores]
    x = torch.tensor(
        [[metrics[revision][source] for source in SOURCE_ORDER] for revision in rows],
        dtype=torch.float64,
    )
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1.0e-8)
    x = (x - mean) / scale
    target_raw = torch.tensor(
        [online_scores[revision] for revision in rows], dtype=torch.float64
    )
    target = (target_raw - target_raw.mean()) / target_raw.std(
        unbiased=False
    ).clamp_min(1.0e-8)

    generator = torch.Generator().manual_seed(seed)
    draws = torch.rand((40000, len(SOURCE_ORDER)), generator=generator, dtype=torch.float64)
    # A Dirichlet-like draw plus a mild max-weight penalty avoids choosing only
    # one task family from a small historical table.
    draws = draws.square()
    draws /= draws.sum(1, keepdim=True).clamp_min(1.0e-12)
    best: tuple[float, torch.Tensor, float, float] | None = None
    for weights in draws:
        predicted = x @ weights
        pred_list = predicted.tolist()
        target_list = target.tolist()
        rank = _rank_correlation(pred_list, target_list)
        rmse = float((predicted - target).square().mean().sqrt())
        penalty = max(float(weights.max()) - 0.60, 0.0)
        objective = rank - 0.08 * rmse - 0.05 * penalty
        if best is None or objective > best[0]:
            best = (objective, weights.clone(), rank, rmse)
    assert best is not None
    _, weights, rank, rmse = best
    predicted_tensor = x @ weights
    centered_prediction = predicted_tensor - predicted_tensor.mean()
    slope = (
        (centered_prediction * (target_raw - target_raw.mean())).sum()
        / centered_prediction.square().sum().clamp_min(1.0e-12)
    )
    intercept = target_raw.mean() - slope * predicted_tensor.mean()
    predicted_points = intercept + slope * predicted_tensor
    predicted = predicted_tensor.tolist()
    return {
        "revisions": rows,
        "source_order": list(SOURCE_ORDER),
        "weights": [float(value) for value in weights],
        "feature_mean": [float(value) for value in mean],
        "feature_scale": [float(value) for value in scale],
        "online_scores": [float(online_scores[revision]) for revision in rows],
        "predicted_z": [float(value) for value in predicted],
        "score_intercept": float(intercept),
        "score_slope": float(slope),
        "predicted_scores": [float(value) for value in predicted_points],
        "mean_absolute_score_error": float(
            (predicted_points - target_raw).abs().mean()
        ),
        "rank_correlation": rank,
        "pairwise_accuracy": _pairwise_accuracy(predicted, target.tolist()),
        "normalized_rmse": rmse,
    }


def manifest(
    attention_seeds: tuple[int, ...] = DEFAULT_ATTENTION_SEEDS,
    linear_seeds: tuple[int, ...] = DEFAULT_LINEAR_SEEDS,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "description": (
            "Fixed-seed random trend set. Public and Qwen tensors are optional "
            "guards and never participate in the fitted trend score."
        ),
        "attention_robust_seeds": list(attention_seeds),
        "linear_robust_seeds": list(linear_seeds),
        "sources": {
            "random_exam": "random_exam_dataset.py / random_trend_data/*.pt",
            "random_robust": "robust_*_benchmark.py fixed seeds",
            "optional_public_guard": "../organizer_sparse/data/{attn,linear}.pt",
            "optional_qwen_guard": "real_model_data/qwen2_5_0_5b.pt",
        },
        "online_results": [
            {
                "revision": revision,
                "score": score,
                "seconds": seconds,
                "status": status,
            }
            for revision, score, seconds, status in ONLINE_RESULTS
        ],
        "case_counts": {
            "attn_random": len(EXAM_ATTENTION_PROFILES) + len(attention_seeds) * len(PROFILES),
            "linear_random": len(EXAM_LINEAR_PROFILES) + len(linear_seeds) * 9,
            "optional_attn_public_guard": 1,
            "optional_linear_public_guard": 1,
            "optional_attn_qwen_guard": 3,
            "optional_linear_qwen_guard": 3,
        },
    }


def _online_score_map() -> dict[str, float]:
    return {revision: float(score) for revision, score, _, _ in ONLINE_RESULTS if score is not None}


def _print_metrics(metrics: dict[str, dict[str, float]], revisions: list[str]) -> None:
    for revision in revisions:
        row = metrics[revision]
        fields = [
            f"{name}={row[name]:+.5f}"
            for name in SOURCE_ORDER
            if name in row
        ]
        for name in ("attn_public", "attn_qwen", "linear_public", "linear_qwen"):
            if name in row:
                fields.append(f"guard:{name}={row[name]:+.5f}")
        print(f"{revision:10s} " + " ".join(fields))


def _predict_candidate(
    calibration: dict[str, Any], row: dict[str, float]
) -> dict[str, float]:
    standardized = [
        (row[name] - mean) / max(scale, 1.0e-12)
        for name, mean, scale in zip(
            calibration["source_order"],
            calibration["feature_mean"],
            calibration["feature_scale"],
        )
    ]
    trend_z = sum(
        value * weight
        for value, weight in zip(standardized, calibration["weights"])
    )
    predicted_score = (
        calibration["score_intercept"]
        + calibration["score_slope"] * trend_z
    )
    return {
        "trend_z": float(trend_z),
        "illustrative_predicted_score": float(predicted_score),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="online_trend_dataset.json")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--refit-existing", action="store_true")
    parser.add_argument("--score")
    parser.add_argument("--include-guards", action="store_true")
    parser.add_argument("--revisions", nargs="+", default=[
        "d75e03a", "5b922c8", "a649209", "def4524", "3c40705",
        "988385e", "be6ffae", "fe4b879", "768a670", "8d88fbc",
        "3e93bd2",
    ])
    parser.add_argument("--attention-seeds", nargs="*", type=int, default=list(DEFAULT_ATTENTION_SEEDS))
    parser.add_argument("--linear-seeds", nargs="*", type=int, default=list(DEFAULT_LINEAR_SEEDS))
    args = parser.parse_args()
    repo = _repo_root()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo / output_path
    existing = (
        json.loads(output_path.read_text(encoding="utf-8"))
        if output_path.exists()
        else None
    )
    result = manifest(tuple(args.attention_seeds), tuple(args.linear_seeds))
    if args.refit_existing:
        if existing is None or "metrics" not in existing:
            raise ValueError("--refit-existing requires an evaluated output file")
        metrics = existing["metrics"]
        revisions = [
            revision for revision in args.revisions if revision in metrics
        ]
        result["metrics"] = metrics
        result["calibration"] = fit_source_weights(
            metrics, revisions, _online_score_map()
        )
    if args.evaluate or args.score:
        cases = materialize_cases(
            repo,
            tuple(args.attention_seeds),
            tuple(args.linear_seeds),
            include_guards=args.include_guards,
        )
        revisions = list(args.revisions) if args.evaluate else [args.score]
        metrics: dict[str, dict[str, float]] = {}
        for revision in revisions:
            started = time.perf_counter()
            solution = load_solution(repo, revision)
            with torch.no_grad():
                metrics[revision] = evaluate_revision(solution, cases)
            print(f"evaluated {revision} in {time.perf_counter() - started:.2f}s")
        _print_metrics(metrics, revisions)
        if args.evaluate:
            result["metrics"] = metrics
            online_scores = _online_score_map()
            result["calibration"] = fit_source_weights(metrics, revisions, online_scores)
            calibration = result["calibration"]
            print(
                "fit rank_corr="
                f"{calibration['rank_correlation']:.4f} "
                "pairwise="
                f"{calibration['pairwise_accuracy']:.4f} "
                "weights="
                + ",".join(
                    f"{name}:{weight:.3f}"
                    for name, weight in zip(
                        calibration["source_order"], calibration["weights"]
                    )
                )
            )
        else:
            if existing is not None:
                result = existing
            row = metrics[args.score]
            candidate = {"metrics": row}
            if "calibration" in result:
                candidate.update(_predict_candidate(result["calibration"], row))
                print(
                    f"trend_z={candidate['trend_z']:+.4f} "
                    "illustrative_score="
                    f"{candidate['illustrative_predicted_score']:.0f}"
                )
            result.setdefault("candidate_scores", {})[args.score] = candidate
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
