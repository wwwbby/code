# HiF4 quantization demo

Huawei algorithm contest demo for converting NVFP4 carrier + block scales into
legal HiF4 parameters for Linear and Attention workloads.

## Files

- `solution.py`: the six public contest APIs and quantization implementation.
- `EXPERIMENT_LOG.md`: current-dataset online results, rejected ideas, runtime
  limits, and promotion gates for future candidates.
- `local_proxy_benchmark.py`: reproducible synthetic data generator and proxy
  scorer. It compares the demo with the paper-style peak conversion baseline.
- `real_model_benchmark.py`: captures weights, activations, and Q/K/V tensors
  from real Qwen layers and evaluates historical revisions on them.
- `attention_alpha_sweep.py`: checks whether calibration-derived Q/K balance
  and V-importance choices generalize across real layers.
- `linear_rank_sweep.py`: measures the Linear low-rank/sweep quality frontier.
- `attention_component_ablation.py`: isolates Q-only and K-only Hessian gains.
- `attention_k_sweep.py`: measures K-Hessian rank, sweep, guard, and token caps.

## Algorithm

For every 64-value HiF4 block, the implementation:

1. decodes the NVFP4 carrier in FP32;
2. starts from the paper-style `max(abs(x)) / 7` E6M2 scale;
3. searches 12 broad and 5 guarded refinement scale candidates;
4. exactly minimizes elementwise squared error over legal level-2 and level-3
   micro-scale choices;
5. rounds and clamps the 64 leaves to legal S1P2 values.

Linear calibration applies reciprocal SmoothQuant-style scaling and a shared
signed H64 transform to weights and activations. Weight scale selection uses
activation second moments. Dynamic activation search is weighted by the
transformed weight's output sensitivity. A rank-32 Hessian pass then keeps each
block's refined global scale fixed and performs one guarded local coordinate
sweep. The larger factor retains more Linear covariance at modest offline cost.

Q and K use reciprocal per-channel smoothing followed by the same signed
orthonormal Hadamard across the full attention head. Non-power-of-two heads
retain the H64 fallback. Calibration builds only the Q covariance needed to
score K error, factors it once to rank 8, and stores the small factors in
`k_state`. Dynamic Q stays on the direct path; dynamic K performs two guarded
local sweeps and accepts blocks with at least 10% covariance-loss reduction.
This K-only design captures the useful part of the timed-out Q+K experiment
without repeatedly factorizing Hessians or refining the much larger Q tensor.
V uses a mild, compressed diagonal importance derived from bounded full/causal
attention statistics and is still quantized only once.

The paired Linear and Q/K transforms are algebraically cancelling, so they
preserve the unquantized Linear output and attention logits exactly apart from
floating-point roundoff.

The submission code performs no file I/O and never computes the prohibited
Linear `A @ W` calibration target.

## Local results

On the organizer's public mini sample:

- official output-format checks: `22/22` passed;
- Linear output NMSE cases: `0.00025638`, `0.00031286`, `0.00028422`,
  `0.00031588`, `0.00028812` (mean `0.00029149`);
- Attention proxy score over the five public tests: `0.30040` versus `0.25944`
  for `b7248c6`;
- combined ten-test proxy score: `5.28822` versus `5.08173` for `b7248c6`;
- full self-check runtime on the development machine: typically `25-29 s`.

On three captured `Qwen2.5-0.5B` layers, the K-only refinement raises mean
full-attention improvement over plain HiF4 from `10.83%` to `71.72%` and causal
improvement from `22.85%` to `66.89%`. Linear rank-32 raises the corresponding
Linear proxy from `40.11%` to `41.82%`.

The local-scale solver algebraically reduces eight hierarchy combinations to
three effective total scales while preserving the original tie breaks. The
earlier `d75e03a` revision measured `15300` points in `261 s`, and `5b922c8`
measured `15600` points in `248 s` on the current contest server. These online
measurements are the runtime baseline, rather than historical public reports
from a different evaluator. Rank-8 refinement retains about `95.5%` of the
previous Hessian-lite public Linear improvement while reducing the full local
self-check time enough to fund covariance-aware Q/K and guarded V refinement.
The later `237b142` Q+K Attention-Hessian experiment timed out on the contest
server. Component ablation found that Q-only refinement provided essentially no
gain, while K-only refinement slightly exceeded Q+K on the captured model. The
last submitted `b7248c6` revision remained at `16000` points; this revision is a
new hidden-set candidate, and the contest server remains authoritative for its
score and runtime.

## Run the proxy benchmark

```bash
python local_proxy_benchmark.py --solution_dir .
```

## Run the organizer self-check

Place the organizer's `self_check.py` and `mini_sample` directory next to this
repository, then run:

```bash
python ../self_check.py --solution_dir . --datasets_dir ../mini_sample
```

Create the submission archive with `solution.py` at the ZIP root:

```bash
python -m zipfile -c solution.zip solution.py
```
