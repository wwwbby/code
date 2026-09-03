from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import torch


NVFP4_VALUES = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
      0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
NVFP4_BOUNDARIES = (NVFP4_VALUES[:-1] + NVFP4_VALUES[1:]) * 0.5
SAMPLE_LENGTHS = (16, 32, 48, 64, 80, 16, 32, 48, 64, 80)
LAYER_INDICES = (0, 4, 8)


def to_nvfp4_pair(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(x.shape)
    if shape[-1] % 16:
        raise ValueError(f"last dimension must be divisible by 16: {shape}")
    blocks = x.float().reshape(*shape[:-1], shape[-1] // 16, 16)
    scale = (blocks.abs().amax(dim=-1) / 6.0).clamp_min(2.0 ** -24)
    normalized = blocks / scale.unsqueeze(-1)
    indices = torch.bucketize(normalized, NVFP4_BOUNDARIES.to(x.device))
    quantized = NVFP4_VALUES.to(x.device)[indices]
    return quantized.reshape(shape).to(torch.bfloat16), scale.to(torch.bfloat16)


def from_hif4(params: dict[str, torch.Tensor], shape: tuple[int, ...]) -> torch.Tensor:
    return (
        params["sign"]
        * params["mant"]
        * params["scale_lv3"]
        * params["scale_lv2"]
        * params["scale_factor"]
    ).reshape(shape).float()


def dequant_pair(solution, pair: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return solution.dequantize_nvfp4(*pair).float()


def standard_quantize(solution, value: torch.Tensor) -> dict[str, torch.Tensor]:
    for name in ("_quantize_hif4_direct", "_quantize_hif4_fast", "_quantize_hif4"):
        function = getattr(solution, name, None)
        if function is not None:
            return function(value)
    raise AttributeError("solution has no plain HiF4 quantizer")


def load_solution(repo: Path, revision: str):
    if revision.startswith("file:"):
        source_path = Path(revision[5:]).resolve()
        source = source_path.read_text(encoding="utf-8")
        label = source_path.stem
    elif revision in ("worktree", "current"):
        source = (repo / "solution.py").read_text(encoding="utf-8")
        label = "worktree"
    else:
        source = subprocess.check_output(
            ["git", "show", f"{revision}:solution.py"], cwd=repo, text=True,
            encoding="utf-8",
        )
        label = revision
    module = types.ModuleType(f"solution_{label.replace('-', '_')}")
    exec(compile(source, f"<{label}:solution.py>", "exec"), module.__dict__)
    return module


def collect_real_tensors(model_dir: Path, output_path: Path) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.eval()
    model.model.layers = torch.nn.ModuleList(
        list(model.model.layers[: max(LAYER_INDICES) + 1])
    )

    paragraph = (
        "Quantization compresses transformer weights and activations while preserving "
        "the output of matrix multiplication and grouped-query attention. Calibration "
        "data reveals correlations, outlier channels, token structure, and the effect "
        "of residual normalization. 量化算法需要同时兼顾误差、泛化能力和运行时间。"
    )
    text = "\n".join(f"{index}: {paragraph}" for index in range(80))
    total_tokens = sum(SAMPLE_LENGTHS)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=total_tokens,
    )
    if int(encoded["input_ids"].shape[1]) < total_tokens:
        raise RuntimeError("the calibration text did not produce enough tokens")

    captured: dict[int, dict[str, torch.Tensor]] = {
        index: {} for index in LAYER_INDICES
    }
    handles = []

    def save_output(layer_index: int, name: str):
        def hook(_module, _inputs, output):
            captured[layer_index][name] = output.detach().squeeze(0).float().cpu()
        return hook

    def save_input(layer_index: int, name: str):
        def hook(_module, inputs, _output):
            captured[layer_index][name] = inputs[0].detach().squeeze(0).float().cpu()
        return hook

    for index in LAYER_INDICES:
        layer = model.model.layers[index]
        handles.extend([
            layer.self_attn.q_proj.register_forward_hook(save_output(index, "q")),
            layer.self_attn.k_proj.register_forward_hook(save_output(index, "k")),
            layer.self_attn.v_proj.register_forward_hook(save_output(index, "v")),
            layer.mlp.up_proj.register_forward_hook(save_input(index, "activation")),
        ])

    started = time.perf_counter()
    with torch.inference_mode():
        model.model(**encoded, use_cache=False)
    forward_seconds = time.perf_counter() - started
    for handle in handles:
        handle.remove()

    config = model.config
    cases = []
    offsets = [0]
    for length in SAMPLE_LENGTHS:
        offsets.append(offsets[-1] + length)
    for index in LAYER_INDICES:
        layer = model.model.layers[index]
        tensors = captured[index]
        linear_pairs = [
            to_nvfp4_pair(tensors["activation"][offsets[i]:offsets[i + 1]])
            for i in range(10)
        ]
        attention_samples = []
        for i in range(10):
            start, end = offsets[i], offsets[i + 1]
            attention_samples.append({
                name: to_nvfp4_pair(tensors[name][start:end])
                for name in ("q", "k", "v")
            })
        cases.append({
            "name": f"Qwen2.5-0.5B/layer-{index}",
            "linear": {
                "weight": to_nvfp4_pair(layer.mlp.up_proj.weight.detach().float().cpu()),
                "calib_activation_list": linear_pairs[:5],
                "test_activation_list": linear_pairs[5:],
            },
            "attention": {
                "q_num_heads": int(config.num_attention_heads),
                "kv_num_heads": int(config.num_key_value_heads),
                "head_dim": int(config.hidden_size // config.num_attention_heads),
                "calib": attention_samples[:5],
                "test": attention_samples[5:],
            },
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "real-model-hif4-v1",
        "model": "Qwen/Qwen2.5-0.5B",
        "forward_seconds": forward_seconds,
        "cases": cases,
    }, output_path)
    print(f"saved {len(cases)} layer cases to {output_path} ({forward_seconds:.2f}s forward)")


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    causal: bool,
) -> torch.Tensor:
    length = q.shape[0]
    qh = q.reshape(length, q_heads, head_dim).transpose(0, 1)
    kh = k.reshape(length, kv_heads, head_dim).transpose(0, 1)
    vh = v.reshape(length, kv_heads, head_dim).transpose(0, 1)
    repeat = q_heads // kv_heads
    kh = kh.repeat_interleave(repeat, dim=0)
    vh = vh.repeat_interleave(repeat, dim=0)
    logits = qh @ kh.transpose(-1, -2) / math.sqrt(head_dim)
    if causal:
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        logits = logits.masked_fill(mask, float("-inf"))
    probability = torch.softmax(logits, dim=-1)
    return (probability @ vh).transpose(0, 1).reshape(length, q_heads * head_dim)


def relative_score(reference: torch.Tensor, baseline: torch.Tensor, candidate: torch.Tensor) -> float:
    baseline_mse = (baseline - reference).square().mean().item()
    candidate_mse = (candidate - reference).square().mean().item()
    return (baseline_mse - candidate_mse) / max(baseline_mse, 1.0e-30)


def evaluate_case(solution, case: dict) -> dict[str, float]:
    linear = case["linear"]
    weight = dequant_pair(solution, linear["weight"])
    calibration = solution.hif4_calibration_and_quantize_weight(
        *linear["weight"], linear["calib_activation_list"]
    )
    player_weight = from_hif4(calibration["weight_params"], tuple(weight.shape))
    standard_weight = from_hif4(standard_quantize(solution, weight), tuple(weight.shape))
    linear_scores = []
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
        linear_scores.append(relative_score(
            reference,
            standard_value @ standard_weight.T,
            player_value @ player_weight.T,
        ))

    attn = case["attention"]
    q_heads = attn["q_num_heads"]
    kv_heads = attn["kv_num_heads"]
    head_dim = attn["head_dim"]
    states = solution.hif4_calibration_attention(
        attn["calib"], q_heads, kv_heads, head_dim
    )
    full_scores, causal_scores = [], []
    for sample in attn["test"]:
        original = {name: dequant_pair(solution, sample[name]) for name in ("q", "k", "v")}
        standard = {
            name: from_hif4(standard_quantize(solution, original[name]), tuple(original[name].shape))
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
                functions[name](*sample[name], heads[name], head_dim, states[f"{name}_state"]),
                tuple(original[name].shape),
            )
            for name in ("q", "k", "v")
        }
        for causal, target in ((False, full_scores), (True, causal_scores)):
            target.append(relative_score(
                attention(**original, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal),
                attention(**standard, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal),
                attention(**player, q_heads=q_heads, kv_heads=kv_heads, head_dim=head_dim, causal=causal),
            ))
    return {
        "linear": sum(linear_scores) / len(linear_scores),
        "attention_full": sum(full_scores) / len(full_scores),
        "attention_causal": sum(causal_scores) / len(causal_scores),
    }


def benchmark(repo: Path, dataset_path: Path, revisions: list[str]) -> None:
    bundle = torch.load(dataset_path, map_location="cpu", weights_only=False)
    print(f"dataset={bundle['model']} cases={len(bundle['cases'])} capture={bundle['forward_seconds']:.2f}s")
    for revision in revisions:
        solution = load_solution(repo, revision)
        started = time.perf_counter()
        results = []
        for case in bundle["cases"]:
            result = evaluate_case(solution, case)
            results.append(result)
            print(
                f"{revision:10s} {case['name']:26s} "
                f"linear={result['linear']:+.4f} "
                f"attn_full={result['attention_full']:+.4f} "
                f"attn_causal={result['attention_causal']:+.4f}"
            )
        means = {
            key: sum(result[key] for result in results) / len(results)
            for key in results[0]
        }
        elapsed = time.perf_counter() - started
        print(
            f"{revision:10s} MEAN linear={means['linear']:+.4f} "
            f"attn_full={means['attention_full']:+.4f} "
            f"attn_causal={means['attention_causal']:+.4f} elapsed={elapsed:.2f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="real_models/Qwen2.5-0.5B")
    parser.add_argument("--dataset", default="real_model_data/qwen2_5_0_5b.pt")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--revisions", nargs="+", default=["a649209", "worktree"])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    dataset = (repo / args.dataset).resolve()
    if args.collect or not dataset.exists():
        collect_real_tensors((repo / args.model_dir).resolve(), dataset)
    benchmark(repo, dataset, args.revisions)


if __name__ == "__main__":
    main()
