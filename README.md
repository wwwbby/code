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
- `robust_linear_benchmark.py`: compares Linear components across nine
  distribution families, public data, and captured real-model data.
- `robust_attention_benchmark.py`: covers MHA/GQA/MQA, head dimensions
  64/128/256, short/long sequences, tails, and Q/K imbalance.
- `random_exam_dataset.py`: materializes a deterministic random exam-like
  dataset with IID, scaled, correlated and sparse-heavy-tail families.
- `online_trend_dataset.py`: replays historical revisions on that random set
  and calibrates a two-component Attention/Linear trend score.
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
V starts from a mild, compressed diagonal importance derived from bounded
full/causal attention statistics. Four nested 4/8/32-token error-exchange
updates and two inexpensive analytic full/causal prefix updates then adjust
only legal mantissas while retaining the chosen scale hierarchy. The prefix
blend remains length-decayed, so the stronger cap affects short sequences most.

The final calibration guard compares each richer Linear and paired-Q/K path
with direct alternatives on small deterministic projections of the supplied
calibration tensors. Linear requires a 2% aggregate improvement plus agreement
on alternating calibration folds. A second Q/K selector can choose rich,
direct, Hadamard-only, Smooth-only, or Smooth+Hadamard and requires a 5%
end-to-end Attention improvement over the incumbent. A rejected rich Linear
path receives activation-energy-weighted Weight codes and a quantized-opponent
Activation correction instead of reverting all the way to plain MSE.

The paired Linear and Q/K transforms are algebraically cancelling, so they
preserve the unquantized Linear output and attention logits exactly apart from
floating-point roundoff.

The submission code performs no file I/O and never materializes the prohibited
Linear `A @ W` calibration target; the Linear guard evaluates its small output
residual directly from activation and weight quantization errors.

## Local results

On the organizer's public mini sample:

- official output-format checks: `22/22` passed;
- Linear output NMSE cases: `0.00025638`, `0.00031286`, `0.00028422`,
  `0.00031588`, `0.00028812` (mean `0.00029149`);
- public relative-quality guards: Linear `0.82433`; Attention
  `0.37584/0.37543` for full/causal evaluation;
- full self-check runtime on the development machine: typically `25-29 s`.

On three captured `Qwen2.5-0.5B` layers, the K-only refinement raises mean
full-attention improvement over plain HiF4 from `10.83%` to `71.72%` and causal
improvement from `22.85%` to `66.89%`. Linear rank-32 raises the corresponding
Linear proxy from `40.11%` to `41.82%`.

On the fixed-seed random trend set, the current candidate moves Attention from
`0.26475` to `0.27735` and Linear from `0.22338` to `0.24089` relative to
`c11d987`. After the measured `c11d987` result was added, the refitted affine
proxy gives only an illustrative `19723`; importantly, it overpredicts the
known `c11d987` result by 938 points. The local score is therefore a rejection
gate, not an online forecast.

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
best verified revision is `c11d987` at `18471` points in `236 s`. The earlier
cross-validated alpha/K candidate `e0a19b0` regressed to `16700` points in
`250 s` despite improving all local proxy groups, so that experiment was
restored before the later verified components were added. On the complete
public-data API run, two paired timings were `37.52/38.56 s` for `c11d987` and
`40.23/39.92 s` for the current candidate, an average increase of 5.4%. A
linear extrapolation from 236 seconds is about 249 seconds, below the 300-second
cutoff but still subject to server variance. The contest server remains the
only authoritative source for score and runtime ordering.

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
