"""Generate deterministic model-shaped random HiF-style HiF datasets.

The public tensors look like large, independently generated test matrices with
several sequence lengths, rather than a contiguous model trace.  This builder
therefore creates multiple statistical families and emits the exact ``linear.pt``
and ``attn.pt`` interface consumed by the organizer self-check.

The tensors are generated from fixed seeds and converted to NVFP4 carriers.
Generated ``*.pt`` files are intentionally git-ignored; the recipe and JSON
manifest are the reproducible dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import torch

from real_model_benchmark import to_nvfp4_pair


# The last sample on both sides reaches 1024 tokens, matching the public
# evaluator's long-sequence regime without making every sample that large.
LENGTHS = (16, 64, 192, 384, 1024, 16, 64, 192, 384, 1024)


@dataclass(frozen=True)
class LinearProfile:
    name: str
    channels: int
    rows: int
    family: str
    channel_spread: float = 0.0
    tail_probability: float = 0.0
    correlation: float = 0.0


@dataclass(frozen=True)
class AttentionProfile:
    name: str
    q_heads: int
    kv_heads: int
    head_dim: int
    family: str
    channel_spread: float = 0.0
    tail_probability: float = 0.0
    q_rms: float = 1.0
    k_rms: float = 1.0
    v_token_correlation: float = 0.0


LINEAR_PROFILES = (
    LinearProfile("iid-d512", 512, 768, "iid"),
    LinearProfile(
        "channel-scaled-d1024", 1024, 768, "channel_scaled",
        channel_spread=0.85,
    ),
    LinearProfile(
        "correlated-d1024", 1024, 512, "correlated",
        correlation=0.82,
    ),
    LinearProfile(
        "sparse-tail-d768", 768, 512, "sparse_tail",
        tail_probability=0.012,
    ),
    LinearProfile(
        "public-like-d1024", 1024, 512, "public_like",
        channel_spread=0.55, tail_probability=0.0015,
    ),
)

ATTENTION_PROFILES = (
    AttentionProfile("gqa-iid-h8-kv2-d128", 8, 2, 128, "iid"),
    AttentionProfile(
        "gqa-scaled-h16-kv2-d128", 16, 2, 128, "channel_scaled",
        channel_spread=0.75,
    ),
    AttentionProfile(
        "mha-tail-h4-d128", 4, 4, 128, "sparse_tail",
        tail_probability=0.01,
    ),
    AttentionProfile(
        "mqa-correlated-h8-d256", 8, 1, 256, "correlated",
        q_rms=1.2, k_rms=1.2, v_token_correlation=0.72,
    ),
)


def _normal(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, generator=generator)


def _unit_rms(value: torch.Tensor, rms: float = 1.0) -> torch.Tensor:
    return value * (rms / value.square().mean().sqrt().clamp_min(1.0e-8))


def _sparse_tail(
    value: torch.Tensor,
    generator: torch.Generator,
    probability: float,
    amplitude: float = 8.0,
) -> torch.Tensor:
    mask = torch.rand(value.shape, generator=generator) < probability
    impulse = torch.randn(value.shape, generator=generator) * amplitude
    return value + mask * impulse


def _ar_channels(value: torch.Tensor, strength: float) -> torch.Tensor:
    output = value.clone()
    output[..., 1:] += strength * value[..., :-1]
    output[..., 2:] += strength * strength * value[..., :-2]
    return output / math.sqrt(1.0 + strength * strength + strength**4)


def _ar_tokens(value: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0.0 or value.shape[0] <= 1:
        return value
    output = value.clone()
    for index in range(1, value.shape[0]):
        output[index] += strength * output[index - 1]
    return output * math.sqrt(max(1.0 - strength * strength, 1.0e-4))


def _channel_scale(
    generator: torch.Generator,
    channels: int,
    spread: float,
) -> torch.Tensor:
    if spread <= 0.0:
        return torch.ones(channels)
    return torch.exp(torch.randn(channels, generator=generator) * spread).clamp(0.15, 6.0)


def _token_modulation(
    generator: torch.Generator,
    length: int,
) -> torch.Tensor:
    # Mixture of smooth and independent token amplitude variation.  This is
    # deliberately generic and does not copy any model activation trace.
    independent = 0.72 + 0.56 * torch.rand((length, 1), generator=generator)
    phase = torch.rand((), generator=generator) * (2.0 * math.pi)
    position = torch.arange(length, dtype=torch.float32)
    smooth = 1.0 + 0.12 * torch.sin(position * (2.0 * math.pi / max(length, 1)) + phase)
    return independent * smooth[:, None]


def _linear_group(profile: LinearProfile, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    channel_scale = _channel_scale(generator, profile.channels, profile.channel_spread)

    if profile.family == "correlated":
        weight = _ar_channels(
            _normal(generator, (profile.rows, profile.channels)),
            profile.correlation,
        )
    else:
        weight = _normal(generator, (profile.rows, profile.channels))
    if profile.family in ("channel_scaled", "public_like"):
        weight = weight / channel_scale.sqrt()
    if profile.family == "sparse_tail":
        weight = _sparse_tail(weight, generator, profile.tail_probability)
    elif profile.family == "public_like":
        weight = _sparse_tail(
            weight, generator, profile.tail_probability * 0.08, 18.0
        )
    weight = _unit_rms(weight)

    def activation(length: int) -> torch.Tensor:
        value = _normal(generator, (length, profile.channels))
        if profile.family in ("channel_scaled", "public_like"):
            value = value * channel_scale
        if profile.family == "correlated":
            value = _ar_channels(value, profile.correlation)
        elif profile.family == "sparse_tail":
            value = _sparse_tail(value, generator, profile.tail_probability)
        elif profile.family == "public_like":
            value = _sparse_tail(
                value, generator, profile.tail_probability, 24.0
            )
        value = value * _token_modulation(generator, length)
        return _unit_rms(value)

    samples = [to_nvfp4_pair(activation(length)) for length in LENGTHS]
    return {
        "key": profile.name,
        "weight": to_nvfp4_pair(weight),
        "calib_activation_list": samples[:5],
        "test_activation_list": samples[5:],
    }


def _attention_tensor(
    generator: torch.Generator,
    length: int,
    heads: int,
    dim: int,
    family: str,
    scale: torch.Tensor,
    rms: float,
    tail_probability: float,
    token_correlation: float = 0.0,
) -> torch.Tensor:
    value = _normal(generator, (length, heads, dim))
    if family == "channel_scaled":
        value = value * scale.reshape(1, heads, dim)
    elif family == "correlated":
        value = _ar_channels(value, 0.78)
    elif family == "sparse_tail":
        value = _sparse_tail(value, generator, tail_probability)
    value = _ar_tokens(value, token_correlation)
    value = value * _token_modulation(generator, length).reshape(length, 1, 1)
    return _unit_rms(value, rms).reshape(length, heads * dim)


def _attention_group(profile: AttentionProfile, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    q_scale = _channel_scale(
        generator, profile.q_heads * profile.head_dim, profile.channel_spread
    )
    k_scale = _channel_scale(
        generator, profile.kv_heads * profile.head_dim, profile.channel_spread
    )
    v_scale = _channel_scale(
        generator, profile.kv_heads * profile.head_dim, 0.5 * profile.channel_spread
    )

    def sample(length: int) -> dict:
        return {
            "q": to_nvfp4_pair(_attention_tensor(
                generator, length, profile.q_heads, profile.head_dim,
                profile.family, q_scale, profile.q_rms,
                profile.tail_probability,
            )),
            "k": to_nvfp4_pair(_attention_tensor(
                generator, length, profile.kv_heads, profile.head_dim,
                profile.family, k_scale, profile.k_rms,
                profile.tail_probability,
            )),
            "v": to_nvfp4_pair(_attention_tensor(
                generator, length, profile.kv_heads, profile.head_dim,
                profile.family, v_scale, 1.0,
                profile.tail_probability, profile.v_token_correlation,
            )),
        }

    samples = [sample(length) for length in LENGTHS]
    return {
        "key": profile.name,
        "q_num_heads": profile.q_heads,
        "kv_num_heads": profile.kv_heads,
        "head_dim": profile.head_dim,
        "attn_type": "full-and-causal-proxy",
        "calib": samples[:5],
        "test": samples[5:],
    }


def build(seed: int = 20260908) -> tuple[list[dict], list[dict]]:
    linear = [
        _linear_group(profile, seed + 1000 + index * 97)
        for index, profile in enumerate(LINEAR_PROFILES)
    ]
    attention_groups = [
        _attention_group(profile, seed + 2000 + index * 97)
        for index, profile in enumerate(ATTENTION_PROFILES)
    ]
    return linear, attention_groups


def _decode(pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    quant, scale = pair
    return (
        quant.float().reshape(*quant.shape[:-1], -1, 16)
        * scale.float()[..., None]
    ).reshape_as(quant)


def _stats(pair: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    value = _decode(pair).reshape(-1, pair[0].shape[-1])
    rms = value.square().mean().sqrt().clamp_min(1.0e-12)
    centered = value - value.mean()
    variance = centered.square().mean().clamp_min(1.0e-12)
    kurtosis = centered.pow(4).mean() / variance.square()
    channel_rms = value.square().mean(0).sqrt().clamp_min(1.0e-12)
    adjacent = (
        (value[:, 1:] * value[:, :-1]).mean()
        / (value[:, 1:].square().mean() * value[:, :-1].square().mean()).sqrt().clamp_min(1.0e-12)
    )
    return {
        "rms": float(rms),
        "peak_over_rms": float(value.abs().max() / rms),
        "kurtosis": float(kurtosis),
        "channel_log2_rms_std": float(torch.log2(channel_rms).std()),
        "adjacent_correlation": float(adjacent),
    }


def summarize(linear: list[dict], attention_groups: list[dict]) -> dict:
    return {
        "linear": {
            group["key"]: {
                "weight": _stats(group["weight"]),
                "activation": _stats(group["test_activation_list"][-1]),
            }
            for group in linear
        },
        "attention": {
            group["key"]: {
                role: _stats(group["test"][-1][role])
                for role in ("q", "k", "v")
            }
            for group in attention_groups
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="random_trend_data")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parent
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = repo / output
    output.mkdir(parents=True, exist_ok=True)
    linear, attention_groups = build(args.seed)
    torch.save(linear, output / "linear.pt")
    torch.save(attention_groups, output / "attn.pt")
    metadata = {
        "schema": 1,
        "seed": args.seed,
        "sample_lengths": list(LENGTHS),
        "linear_profiles": [asdict(profile) for profile in LINEAR_PROFILES],
        "attention_profiles": [asdict(profile) for profile in ATTENTION_PROFILES],
        "statistics": summarize(linear, attention_groups),
    }
    with (output / "manifest.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(
        f"wrote {len(linear)} Linear and {len(attention_groups)} Attention "
        f"groups to {output}"
    )


if __name__ == "__main__":
    main()
