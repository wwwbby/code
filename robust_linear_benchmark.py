from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from real_model_benchmark import (
    dequant_pair,
    from_hif4,
    load_solution,
    relative_score,
    standard_quantize,
    to_nvfp4_pair,
)


def _normal(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, generator=generator)


def _student_t(
    generator: torch.Generator,
    shape: tuple[int, ...],
    degrees: int = 3,
) -> torch.Tensor:
    numerator = _normal(generator, shape)
    denominator = sum(
        _normal(generator, shape).square() for _ in range(degrees)
    )
    return numerator / (denominator / float(degrees)).sqrt().clamp_min(0.15)


def _ar_correlate(x: torch.Tensor, strength: float) -> torch.Tensor:
    y = x.clone()
    y[..., 1:] += strength * x[..., :-1]
    y[..., 2:] += (strength * strength) * x[..., :-2]
    return y / math.sqrt(1.0 + strength**2 + strength**4)


def _make_case(
    name: str,
    generator: torch.Generator,
    channels: int,
    rows: int,
    family: str,
) -> dict:
    token_count = 32

    if family == "gaussian":
        weight = _normal(generator, (rows, channels))

        def activation() -> torch.Tensor:
            return _normal(generator, (token_count, channels))

    elif family == "heavy_tail":
        weight = _student_t(generator, (rows, channels))

        def activation() -> torch.Tensor:
            return _student_t(generator, (token_count, channels))

    elif family == "weight_heavy":
        weight = _student_t(generator, (rows, channels))

        def activation() -> torch.Tensor:
            return _normal(generator, (token_count, channels))

    elif family == "activation_heavy":
        weight = _normal(generator, (rows, channels))

        def activation() -> torch.Tensor:
            return _student_t(generator, (token_count, channels))

    elif family == "correlated_weight_heavy":
        weight = _student_t(generator, (rows, channels))

        def activation() -> torch.Tensor:
            return _ar_correlate(
                _normal(generator, (token_count, channels)), 0.85
            )

    elif family == "heteroscedastic":
        order = torch.randperm(channels, generator=generator)
        scales = torch.logspace(-1.25, 1.25, channels)[order]
        weight = _normal(generator, (rows, channels)) / scales.sqrt()

        def activation() -> torch.Tensor:
            return _normal(generator, (token_count, channels)) * scales

    elif family == "correlated":
        weight = _ar_correlate(_normal(generator, (rows, channels)), 0.75)

        def activation() -> torch.Tensor:
            return _ar_correlate(
                _normal(generator, (token_count, channels)), 0.85
            )

    elif family == "sparse_outlier":
        weight = _normal(generator, (rows, channels))
        weight_mask = torch.rand((rows, channels), generator=generator) < 0.01
        weight += weight_mask * _normal(generator, (rows, channels)) * 12.0

        def activation() -> torch.Tensor:
            value = _normal(generator, (token_count, channels))
            mask = torch.rand(value.shape, generator=generator) < 0.01
            return value + mask * _normal(generator, tuple(value.shape)) * 12.0

    elif family == "low_rank":
        rank = 8
        loading = _normal(generator, (rank, channels)) / math.sqrt(rank)
        weight = (
            _normal(generator, (rows, rank)) @ loading
            + 0.25 * _normal(generator, (rows, channels))
        )

        def activation() -> torch.Tensor:
            return (
                _normal(generator, (token_count, rank)) @ loading
                + 0.25 * _normal(generator, (token_count, channels))
            )

    else:
        raise ValueError(f"unknown family: {family}")

    return {
        "name": name,
        "weight": to_nvfp4_pair(weight),
        "calibration": [to_nvfp4_pair(activation()) for _ in range(3)],
        "test": [to_nvfp4_pair(activation()) for _ in range(3)],
    }


def make_cases(seed: int = 20260904) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    specifications = (
        ("iid-d256", 256, 128, "gaussian"),
        ("heavy-tail-d256", 256, 128, "heavy_tail"),
        ("weight-heavy-d768", 768, 128, "weight_heavy"),
        ("activation-heavy-d768", 768, 128, "activation_heavy"),
        ("corr-weight-heavy-d768", 768, 128, "correlated_weight_heavy"),
        ("channel-scale-d576", 576, 128, "heteroscedastic"),
        ("correlated-d512", 512, 128, "correlated"),
        ("sparse-outlier-d1024", 1024, 96, "sparse_outlier"),
        ("low-rank-d896", 896, 128, "low_rank"),
    )
    return [
        _make_case(name, generator, channels, rows, family)
        for name, channels, rows, family in specifications
    ]


def evaluate(solution, case: dict) -> float:
    weight = dequant_pair(solution, case["weight"])
    calibrated = solution.hif4_calibration_and_quantize_weight(
        *case["weight"], case["calibration"]
    )
    standard_weight = from_hif4(
        standard_quantize(solution, weight), tuple(weight.shape)
    )
    player_weight = from_hif4(
        calibrated["weight_params"], tuple(weight.shape)
    )
    scores = []
    for pair in case["test"]:
        activation = dequant_pair(solution, pair)
        reference = activation @ weight.T
        standard_activation = from_hif4(
            standard_quantize(solution, activation), tuple(activation.shape)
        )
        player_activation = from_hif4(
            solution.hif4_dynamic_quantize_activation(
                *pair, calibrated["activation_state"]
            ),
            tuple(activation.shape),
        )
        scores.append(
            relative_score(
                reference,
                standard_activation @ standard_weight.T,
                player_activation @ player_weight.T,
            )
        )
    return sum(scores) / len(scores)


def calibration_statistics(solution, case: dict) -> dict[str, float]:
    weight = dequant_pair(solution, case["weight"])
    activations = torch.cat(
        [dequant_pair(solution, pair) for pair in case["calibration"]], dim=0
    )
    activation_rms = activations.square().mean(dim=0).sqrt().clamp_min(1.0e-8)
    weight_rms = weight.square().mean(dim=0).sqrt().clamp_min(1.0e-8)
    smooth = solution._smooth_scale(
        activations.abs().amax(dim=0),
        weight.abs().amax(dim=0),
        solution._LINEAR_SMOOTH_ALPHA,
    )
    smooth_spread = torch.log2(smooth).std()
    activation_tail = (
        activations.abs().amax(dim=0) / activation_rms
    ).median()
    weight_tail = (weight.abs().amax(dim=0) / weight_rms).median()

    transformed = solution._apply_block_hadamard(activations / smooth)
    blocks = transformed.reshape(
        -1, transformed.shape[-1] // solution._HIF4_BLOCK_SIZE,
        solution._HIF4_BLOCK_SIZE,
    )
    covariance = torch.einsum("nbx,nby->bxy", blocks, blocks)
    covariance /= max(int(blocks.shape[0]), 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    trace = eigenvalues.sum(dim=-1).clamp_min(1.0e-12)
    rank8 = (eigenvalues[:, -8:].sum(dim=-1) / trace).mean()
    rank32 = (eigenvalues[:, -32:].sum(dim=-1) / trace).mean()
    diagonal_energy = torch.diagonal(covariance, dim1=-2, dim2=-1).square().sum(-1)
    total_energy = covariance.square().sum(dim=(-1, -2)).clamp_min(1.0e-12)
    off_diagonal = (1.0 - diagonal_energy / total_energy).mean()
    return {
        "smooth_spread": float(smooth_spread),
        "activation_tail": float(activation_tail),
        "weight_tail": float(weight_tail),
        "rank8": float(rank8),
        "rank32": float(rank32),
        "off_diagonal": float(off_diagonal),
    }


def disable_linear_hessian(solution) -> None:
    def keep_baseline(_values, _hessian, baseline):
        return baseline

    solution._quantize_hif4_low_rank_hessian = keep_baseline


def _learned_reflectors(
    solution,
    activations: list[torch.Tensor],
    channels: int,
    rank: int,
) -> torch.Tensor:
    rows = torch.cat([value.reshape(-1, channels) for value in activations], dim=0)
    blocks = rows.reshape(-1, channels // 64, 64)
    covariance = torch.einsum("nbx,nby->bxy", blocks, blocks)
    covariance /= max(int(blocks.shape[0]), 1)
    source = torch.linalg.eigh(covariance).eigenvectors[..., -rank:].flip(-1)

    basis = torch.eye(64, dtype=torch.float32)[:rank]
    target = solution._apply_block_hadamard(basis).T
    reflectors = []
    for index in range(rank):
        current = source[..., index]
        for reflector in reflectors:
            current = current - 2.0 * (
                current * reflector
            ).sum(dim=-1, keepdim=True) * reflector
        destination = target[:, index].expand_as(current)
        destination = torch.where(
            (current * destination).sum(dim=-1, keepdim=True) < 0.0,
            -destination,
            destination,
        )
        reflector = current - destination
        reflector /= reflector.norm(dim=-1, keepdim=True).clamp_min(1.0e-8)
        reflectors.append(reflector)
    return torch.stack(reflectors, dim=1).contiguous()


def _apply_reflectors(value: torch.Tensor, reflectors: torch.Tensor) -> torch.Tensor:
    original_shape = tuple(value.shape)
    blocks = value.reshape(-1, reflectors.shape[0], 64)
    for index in range(reflectors.shape[1]):
        vector = reflectors[:, index, :].unsqueeze(0)
        blocks = blocks - 2.0 * (blocks * vector).sum(
            dim=-1, keepdim=True
        ) * vector
    return blocks.reshape(original_shape)


def install_learned_rotation(solution, rank: int, use_hessian: bool) -> None:
    def calibrate(weight_quant, weight_scale, calibration):
        weight = solution._dequantize_nvfp4_fp32(weight_quant, weight_scale)
        channels = int(weight.shape[-1])
        activations = [
            solution._dequantize_nvfp4_fp32(quant, scale)
            for quant, scale in calibration
        ]
        activation_peak = torch.stack(
            [value.abs().amax(dim=0) for value in activations]
        ).amax(dim=0)
        smooth = solution._smooth_scale(
            activation_peak,
            weight.abs().amax(dim=0),
            solution._LINEAR_SMOOTH_ALPHA,
        )
        scaled_activations = [value / smooth for value in activations]
        reflectors = _learned_reflectors(
            solution, scaled_activations, channels, rank
        )
        transformed_weight = _apply_reflectors(weight * smooth, reflectors)
        transformed_activations = [
            _apply_reflectors(value, reflectors) for value in scaled_activations
        ]
        activation_second = torch.stack(
            [value.square().mean(dim=0) for value in transformed_activations]
        ).mean(dim=0).clamp_min(1.0e-8)
        baseline = solution._quantize_hif4(transformed_weight, activation_second)
        if use_hessian:
            weight_params = solution._quantize_hif4_low_rank_hessian(
                transformed_weight,
                solution._build_block_hessian_reg(
                    transformed_activations, channels
                ),
                baseline,
            )
        else:
            weight_params = baseline
        importance = transformed_weight.square().sum(dim=0).clamp_min(1.0e-8)
        return {
            "weight_params": weight_params,
            "activation_state": {
                "smooth": smooth.cpu().contiguous(),
                "importance": importance.cpu().contiguous(),
                "reflectors": reflectors.cpu().contiguous(),
            },
        }

    def dynamic(activation_quant, activation_scale, state):
        activation = solution._dequantize_nvfp4_fp32(
            activation_quant, activation_scale
        )
        transformed = _apply_reflectors(
            activation / state["smooth"], state["reflectors"]
        )
        return solution._quantize_hif4(transformed, state["importance"])

    solution.hif4_calibration_and_quantize_weight = calibrate
    solution.hif4_dynamic_quantize_activation = dynamic


def install_linear_ablation(solution, mode: str) -> None:
    """Install a calibration path that isolates one Linear mechanism."""

    if mode == "current":
        return
    if mode in {"learned1", "learned2", "learned2-hessian"}:
        install_learned_rotation(
            solution,
            rank=1 if mode == "learned1" else 2,
            use_hessian=mode.endswith("hessian"),
        )
        return
    if mode not in {"direct", "weighted", "smooth", "rotate", "smooth-rotate"}:
        raise ValueError(f"unknown mode: {mode}")

    use_weighting = mode != "direct"
    use_smooth = mode in {"smooth", "smooth-rotate"}
    use_rotation = mode in {"rotate", "smooth-rotate"}

    def transform(value: torch.Tensor) -> torch.Tensor:
        return solution._apply_block_hadamard(value) if use_rotation else value

    def calibrate(weight_quant, weight_scale, calibration):
        weight = solution._dequantize_nvfp4_fp32(weight_quant, weight_scale)
        channels = int(weight.shape[-1])
        activations = [
            solution._dequantize_nvfp4_fp32(quant, scale)
            for quant, scale in calibration
        ]
        if use_smooth:
            activation_peak = torch.stack(
                [value.abs().amax(dim=0) for value in activations]
            ).amax(dim=0)
            smooth = solution._smooth_scale(
                activation_peak,
                weight.abs().amax(dim=0),
                solution._LINEAR_SMOOTH_ALPHA,
            )
        else:
            smooth = torch.ones(channels, dtype=torch.float32)

        transformed_weight = transform(weight * smooth)
        transformed_activations = [
            transform(value / smooth) for value in activations
        ]
        if use_weighting:
            activation_second = torch.stack(
                [value.square().mean(dim=0) for value in transformed_activations]
            ).mean(dim=0).clamp_min(1.0e-8)
            activation_importance = transformed_weight.square().sum(dim=0)
            activation_importance = activation_importance.clamp_min(1.0e-8)
        else:
            activation_second = None
            activation_importance = torch.ones(channels, dtype=torch.float32)

        state = {
            "smooth": smooth.cpu().contiguous(),
            "importance": activation_importance.cpu().contiguous(),
            "rotation": use_rotation,
            "weighted": use_weighting,
        }
        return {
            "weight_params": solution._quantize_hif4(
                transformed_weight, activation_second
            ),
            "activation_state": state,
        }

    def dynamic(activation_quant, activation_scale, state):
        activation = solution._dequantize_nvfp4_fp32(
            activation_quant, activation_scale
        )
        transformed = transform(activation / state["smooth"])
        importance = state["importance"] if state["weighted"] else None
        return solution._quantize_hif4(transformed, importance)

    solution.hif4_calibration_and_quantize_weight = calibrate
    solution.hif4_dynamic_quantize_activation = dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revisions", nargs="+", default=["worktree"])
    parser.add_argument("--include-no-hessian", action="store_true")
    parser.add_argument("--real-dataset")
    parser.add_argument("--public-linear")
    parser.add_argument("--show-statistics", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "current", "direct", "weighted", "smooth", "rotate", "smooth-rotate",
            "learned1", "learned2", "learned2-hessian",
        ),
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent
    if args.public_linear:
        group = torch.load(
            repo / args.public_linear, map_location="cpu", weights_only=False
        )[0]
        cases = [{
            "name": str(group.get("key", "public-linear")),
            "weight": group["weight"],
            "calibration": group["calib_activation_list"],
            "test": group["test_activation_list"],
        }]
    elif args.real_dataset:
        bundle = torch.load(
            repo / args.real_dataset, map_location="cpu", weights_only=False
        )
        cases = [
            {
                "name": case["name"],
                "weight": case["linear"]["weight"],
                "calibration": case["linear"]["calib_activation_list"],
                "test": case["linear"]["test_activation_list"],
            }
            for case in bundle["cases"]
        ]
    else:
        cases = make_cases()
    if args.show_statistics:
        solution = load_solution(repo, args.revisions[0])
        for case in cases:
            values = calibration_statistics(solution, case)
            rendered = " ".join(f"{key}={value:.3f}" for key, value in values.items())
            print(f"{case['name']:24s} {rendered}")
        return
    if args.modes:
        configurations = [
            (revision, mode) for revision in args.revisions for mode in args.modes
        ]
    else:
        configurations = [(revision, "current") for revision in args.revisions]
        if args.include_no_hessian:
            configurations += [
                (revision, "current-no-H") for revision in args.revisions
            ]

    for revision, mode in configurations:
        solution = load_solution(repo, revision)
        if mode == "current-no-H":
            disable_linear_hessian(solution)
        else:
            install_linear_ablation(solution, mode)
        suffix = "" if mode == "current" else f"-{mode}"
        label = f"{revision}{suffix}"
        started = time.perf_counter()
        scores = []
        for case in cases:
            score = evaluate(solution, case)
            scores.append(score)
            print(f"{label:20s} {case['name']:24s} linear={score:+.5f}")
        elapsed = time.perf_counter() - started
        print(
            f"{label:20s} MEAN={sum(scores) / len(scores):+.5f} "
            f"WORST={min(scores):+.5f} elapsed={elapsed:.2f}s"
        )


if __name__ == "__main__":
    main()
