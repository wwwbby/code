# HiF4 quantization demo

Huawei algorithm contest demo for converting NVFP4 carrier + block scales into
legal HiF4 parameters for Linear and Attention workloads.

## Files

- `solution.py`: the six public contest APIs and quantization implementation.
- `local_proxy_benchmark.py`: reproducible synthetic data generator and proxy
  scorer. It compares the demo with the paper-style peak conversion baseline.

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
transformed weight's output sensitivity. A budgeted Hessian-lite pass then
keeps each block's refined global scale fixed and performs two guarded local
coordinate sweeps; this captures most of the block-covariance benefit without
the full 17-scale Hessian search that timed out.

Q and K use reciprocal per-channel smoothing with a zero-overhead `0.4375`
balance, followed by the same signed H64 transform inside every attention
head. V stays in its original basis because the interface has no
inverse-transform hook and uses the same fast search.

The paired Linear and Q/K transforms are algebraically cancelling, so they
preserve the unquantized Linear output and attention logits exactly apart from
floating-point roundoff.

The submission code performs no file I/O and never computes the prohibited
Linear `A @ W` calibration target.

## Local results

On the organizer's public mini sample:

- official output-format checks: `22/22` passed;
- Linear output NMSE cases: `0.00025313`, `0.00030953`, `0.00028150`,
  `0.00031212`, `0.00028546` (mean `0.00028835`);
- mixed causal/full Attention output NMSE: `0.00470634`;
- full self-check runtime on the development machine: about `27 s`.

The local-scale solver algebraically reduces eight hierarchy combinations to
three effective total scales while preserving the original tie breaks. The
earlier `d75e03a` revision measured `15300` points in `261 s` on the current
contest server; this online measurement is the runtime baseline, rather than
historical public score reports from a different evaluator. Hessian-lite
reduces the public Linear output NMSE by another `19.5%`, while the revised Q/K
balance reduces the public Attention proxy NMSE by `7.14%` without adding a
new runtime pass. A fresh hidden-set submission is still authoritative for
this revision.

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
