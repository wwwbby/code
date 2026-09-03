# HiF4 quantization demo

Huawei algorithm contest demo for converting NVFP4 carrier + block scales into
legal HiF4 parameters for Linear and Attention workloads.

## Files

- `solution.py`: the six public contest APIs and quantization implementation.
- `local_proxy_benchmark.py`: reproducible synthetic data generator and proxy
  scorer. It compares the demo with the paper-style peak conversion baseline.
- `real_model_benchmark.py`: captures weights, activations, and Q/K/V tensors
  from real Qwen layers and evaluates historical revisions on them.
- `attention_alpha_sweep.py`: checks whether calibration-derived Q/K balance
  choices generalize across real layers.

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
transformed weight's output sensitivity. A rank-8 Hessian pass then keeps each
block's refined global scale fixed and performs one guarded local coordinate
sweep. Its exact diagonal plus eight leading covariance directions retain most
of the block-Hessian benefit without the expensive dense 64x64 updates.

Q and K use reciprocal per-channel smoothing followed by the same signed H64
transform inside every attention head. Balanced Q/K groups retain the public
`0.4375` exponent; when calibration K RMS exceeds Q RMS by more than 2x, the
exponent falls to `0.25`. This two-regime rule is computed from two dot products
inside the existing calibration pass and does not add dynamic candidates.
Q and K then use the original direct HiF4 search, while V stays on the fast
direct NVFP4-to-HiF4 path. Attention covariance/Hessian and probability-weighted
V experiments were removed after `237b142` exceeded the contest time limit.

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
- mixed causal/full Attention output NMSE: `0.00470634`;
- full self-check runtime on the development machine: `22.6 s`.

On three captured `Qwen2.5-0.5B` layers, the adaptive Q/K exponent raises the
mean full-attention improvement over plain HiF4 from `-2.40%` to `10.83%` and
the causal-attention improvement from `0.71%` to `22.85%`. The public tensor has
K/Q RMS `1.09`, so it keeps `0.4375` and remains bitwise equal to `a649209`.

The local-scale solver algebraically reduces eight hierarchy combinations to
three effective total scales while preserving the original tie breaks. The
earlier `d75e03a` revision measured `15300` points in `261 s`, and `5b922c8`
measured `15600` points in `248 s` on the current contest server. These online
measurements are the runtime baseline, rather than historical public reports
from a different evaluator. Rank-8 refinement retains about `95.5%` of the
previous Hessian-lite public Linear improvement while reducing the full local
self-check time enough to fund covariance-aware Q/K and guarded V refinement.
The later `237b142` Attention-Hessian experiment timed out on the contest
server, so it is not a usable baseline. This revision returns to the measured
`a649209` performance envelope (`243 s`, `16000` points) and adds only the two
calibration dot products needed by the adaptive exponent. A fresh hidden-set
submission is still authoritative for its score and runtime.

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
