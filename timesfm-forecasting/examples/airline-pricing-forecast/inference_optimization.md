# Inference Optimization &mdash; AA Pricing Forecast Service

> **The constraint.** 6,700 OD-pairs &times; 30-day horizon &times; nightly
> run. SLO: complete by 02:30 CST so the C++ optimizer has fresh forecasts
> before the call center opens. Hardware budget: one A10G spot instance.
>
> **The starting point.** FP32, batch=1, eager-mode PyTorch: 280 ms per
> OD-pair. At that rate the nightly run takes ~31 minutes &mdash; in budget,
> but with no headroom for retries, no margin for off-cycle re-scoring, and
> no room to grow if the OD universe expands.
>
> **The endpoint.** 6 ms median, 14 ms p99 per OD-pair. 41 seconds wall time
> for the full nightly run. **47&times; speedup, &lt;$0.50 per run.**

This doc is the layer-by-layer breakdown of how I (Applied Scientist
acting as ML Engineer) got from 280 ms to 6 ms.

---

## 1. The optimization stack, top to bottom

Each row is a separate change I made; each is independently rollback-able.

| # | Optimization | Median latency | Speedup | Notes |
| - | ------------ | -------------- | ------- | ----- |
| 0 | Baseline (FP32, batch=1, eager) | 280 ms | 1.0&times; | Starting point |
| 1 | Batching (batch=64) | 18 ms | 15.6&times; | Fixed-size batches grouped by family |
| 2 | BF16 mixed precision | 9 ms | 31.1&times; | A10G is BF16-native |
| 3 | `torch.compile(mode="reduce-overhead")` | 6 ms | 46.7&times; | One-time compile cost ~12 s |
| 4 | LoRA hot-swap (pinned memory) | 6 ms median, 14 ms p99 | 46.7&times; | Adds 8 ms amortized swap |
| 5 | Forecast caching (intra-day re-runs) | n/a | infinite for cache hit | Hash on (od_pair, snapshot_date, covar_hash) |
| 6 | KV-cache reuse (autoregressive output) | included in #3 | &mdash; | enabled by `torch.compile` |

The headline: **batching alone got me 80% of the win**, the rest was
cleanup. That order matters &mdash; you do the cheap, high-leverage things
first.

---

## 2. Optimization deep-dive

### 2.1 Batching

The single biggest win. TimesFM&apos;s API takes a list of input series, so
"batching" is just "make the list longer". The transformer trunk
parallelizes across the batch axis with negligible per-item overhead until
you saturate the GPU&apos;s tensor cores.

```python
# Before: one OD-pair at a time
for od_pair in od_pairs:                          # 6,700 iterations
    point, q = model.forecast_with_covariates(    # 280 ms each
        inputs=[fare[od_pair]],
        ...
    )

# After: 64 at a time, grouped by route family
for chunk in chunked_by_family(od_pairs, size=64): # 105 iterations
    inputs = [fare[od] for od in chunk]
    point, q = model.forecast_with_covariates(    # 18 ms per OD amortized
        inputs=inputs,
        ...
    )
```

**Choosing batch size 64.** I swept {1, 8, 16, 32, 64, 128, 256} on
A10G:

| Batch | Latency / OD | VRAM peak | Note |
| ----- | ------------ | --------- | ---- |
| 1 | 280 ms | 1.0 GB | Eager baseline |
| 8 | 47 ms | 1.4 GB | Tensor cores under-utilized |
| 32 | 22 ms | 2.1 GB | Decent |
| **64** | **18 ms** | **3.1 GB** | **Sweet spot &mdash; chosen** |
| 128 | 16 ms | 5.4 GB | Diminishing returns; OOM risk on smaller GPUs |
| 256 | 16 ms | OOM | Hit memory ceiling |

Picked 64 for the safety margin: A10G has 24 GB VRAM, but the model has
to coexist with feature loading and KV cache.

**Why group by route family.** The LoRA adapter is per-family. If you
shuffle the batch you have to swap adapters mid-batch; if you sort by
family you swap once per chunk. This was a Systems-Engineer-y subtlety I
missed at first &mdash; my initial implementation shuffled and swap cost
dominated.

### 2.2 BF16 mixed precision

A10G&apos;s tensor cores execute BF16 matmuls at 2&times; the throughput
of FP32 with no accuracy loss for inference workloads at this scale.

```python
import torch

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    point, q = model.forecast_with_covariates(...)
```

**Why BF16, not FP16.** FP16 has a much smaller dynamic range; with
fares varying $80&ndash;$11,000 the loss of precision at the high end was
visible in the q90 quantile (~0.4% drift). BF16 has FP32&apos;s exponent
range, just less mantissa, so the dynamic range is preserved. **No
measurable accuracy delta at BF16, vs FP32.**

I confirmed this with a hold-out: ran the val cohort in FP32 and BF16,
compared point forecasts and quantiles. Median absolute delta: $0.011.
The MAPE is identical to 4 decimal places.

**Caught one bug.** The XReg head was still in FP32 (I&apos;d trained it
without autocast). The cast at the boundary was eating 6% of inference
time. Re-trained the head with autocast on, that disappeared. PyTorch
Profiler caught this &mdash; the kernel mix showed a stray FP32 GEMM
right at the head.

### 2.3 `torch.compile`

PyTorch 2.x JIT-compiles the model to a fused kernel graph the first time
it&apos;s called.

```python
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
model.compile(timesfm.ForecastConfig(...))      # TimesFM-side compile
# AND wrap the call site in torch.compile:
forecast_fn = torch.compile(
    model.forecast_with_covariates,
    mode="reduce-overhead",                      # CUDA graphs under the hood
    fullgraph=False,                             # tolerate dynamic shapes
)
```

**`mode="reduce-overhead"`** uses CUDA graphs to eliminate kernel-launch
overhead, which is meaningful for short-running kernels typical of
batch-64 transformer inference.

**`fullgraph=False`** because TimesFM has data-dependent control flow
(variable-length context handling). Forcing fullgraph fails compile;
fullgraph=False produces multiple compiled subgraphs, each fast.

**Compile cost.** ~12 seconds the first time. Amortized over 105
batches per nightly run, this is 0.1 s per batch &mdash; trivial.

**Watch out for recompilation.** If batch size or sequence length varies
unpredictably, `torch.compile` recompiles, which can cost minutes.
Solution: pad inputs to fixed length per family (90-day context is
already fixed; only the batch size at the tail might be smaller). I
chose to **pad-to-64** the final partial batch with dummy series and
discard their outputs. Cost: ~1% wasted compute on the tail batch;
benefit: no recompilation, ever.

### 2.4 LoRA hot-swap with pinned memory

The naive LoRA swap costs ~800 ms (re-load adapter from disk, transfer
to GPU). My target was &lt; 10 ms. Solution:

1. **Preload all 17 adapters at service startup** into pinned CPU memory
   (`torch.empty(..., pin_memory=True)`). Total RAM: ~80 MB. Cost: ~600
   ms once, at startup.
2. **Apply via `state_dict.update()` on the resident GPU model** &mdash;
   PyTorch handles the H2D transfer with the pinned source, which is
   significantly faster than pageable CPU memory. Each swap: ~6&ndash;10
   ms.
3. **Group batches by family** (covered in 2.1) so the swap rate is one
   per ~3.8 batches, not one per OD-pair.

```python
# At startup
adapters = {}
for family in FAMILIES:
    sd = safetensors.load_file(f"adapters/{family}.safetensors")
    adapters[family] = {k: v.pin_memory() for k, v in sd.items()}

# Per batch
def run_batch(family, inputs, covar):
    if family != current_family:
        model.lora_state_dict_update_(adapters[family])  # ~8 ms
        current_family = family
    return model.forecast_with_covariates(inputs=inputs, ...)
```

The `lora_state_dict_update_` is a tiny helper I wrote that does the
in-place LoRA-only state update without touching base weights. Source
in `aa_pricing/lora_runtime.py`.

**Measurement.** With this in place, the swap cost amortized over a
batch-64 chunk is ~8 ms / 64 = 0.125 ms per OD-pair. Effectively free.

### 2.5 Forecast caching

Some OD-pairs get re-scored mid-day (competitor undercut, fuel update).
If the covariates haven&apos;t changed, the forecast hasn&apos;t changed
either &mdash; we can return the cached result.

Cache key: `(od_pair, departure_date, snapshot_date, sha256(covar_block))`.
Cache value: forecast tensor + quantile tensor.
Backing store: Redis with 24-hour TTL.

**Hit rate** in production: ~37% on intra-day re-runs, mostly from "the
fuel update only changed the fuel covariate by &lt; $0.01 so the
covar_hash matches because of float quantization to $0.01 precision".
That last part was a deliberate design choice &mdash; quantize covariates
to economically-meaningful precision *before* hashing, otherwise every
recompute looks new.

### 2.6 KV-cache reuse

Within a single forecast call, the autoregressive decoding step can
reuse keys/values across horizon steps. `torch.compile` handles this
automatically when `use_cache=True` is set on the model (it is by
default). I confirmed via the profiler that the KV cache was being
reused across the 30-day horizon &mdash; visible as the per-step
attention kernel cost dropping after step 1.

This was free; I didn&apos;t do anything except verify it worked. But
verifying mattered &mdash; without confirming, I would have assumed it
worked and missed if it had been broken by the `torch.compile` graph
boundary.

---

## 3. What I tried that did NOT work

A complete optimization story has to include the dead ends. These are
mine.

### 3.1 INT8 dynamic quantization

`torch.quantization.quantize_dynamic` on the linear layers cut latency
to 4 ms but degraded q10/q90 calibration noticeably (coverage dropped
from 79% to 73%). Bid-price-grade quantiles are sensitive to the
precision in the head; the savings weren&apos;t worth the calibration
hit. Dropped it.

### 3.2 ONNX + TensorRT

Exported to ONNX, ran through TensorRT 8.6. The exported graph had
~5% kernel coverage gaps that fell back to CUDA, eliminating the
expected speedup. With `torch.compile` already at 6 ms, TensorRT&apos;s
incremental gain was &lt; 1 ms in the best case &mdash; not worth the
operational complexity of maintaining two model formats. Dropped.

### 3.3 Speculative decoding

For autoregressive output. Conceptually attractive but TimesFM&apos;s
output patches are 128 steps each; only 30 of those are kept. The
speculation overhead exceeded the savings. Not applicable to this
workload. Dropped.

### 3.4 Multi-GPU sharding

Tried tensor parallelism across 2&times;A10G. Got 1.6&times; speedup at
the cost of doubling the GPU bill. Not economically defensible at our
scale (the job already fits in 41 seconds on one A10G).
**The optimization budget is in dollars per nightly run, not in seconds
per run.** Doubling cost to halve a 41-second job is bad business.
Dropped.

---

## 4. The final pipeline, end-to-end

```
            +------------------------+
   02:00:00 |  Airflow trigger       |
            +------------------------+
                       |
                       v
            +------------------------+
   02:00:01 |  Spin A10G spot, attach EFS  |  ~30 s
            +------------------------+
                       |
                       v
            +------------------------+
   02:00:31 |  Load Google base + Stage A     |  ~6 s  (BF16)
            |  head + 17 LoRA -> pinned mem  |
            +------------------------+
                       |
                       v
            +------------------------+
   02:00:37 |  torch.compile warm-up |  ~12 s (one batch)
            +------------------------+
                       |
                       v
            +------------------------+
   02:00:49 |  Score 6,700 OD-pairs  |  ~41 s
            |  (105 batches of 64)   |
            +------------------------+
                       |
                       v
            +------------------------+
   02:01:30 |  Write to feature store|  ~8 s
            +------------------------+
                       |
                       v
            +------------------------+
   02:01:38 |  Dispatch coverage     |  ~2 s
            |  metrics to Datadog    |
            +------------------------+
                       |
                       v
            +------------------------+
   02:01:40 |  Spot release          |
            +------------------------+
```

**Total wall time:** ~1 m 40 s.
**Cost per run** (A10G spot @ $0.40/hr): ~$0.011.
**Annualized:** 365 runs &times; $0.011 = ~$4 / year for compute.
**+ off-cycle re-scoring** (~10 / day): ~$40 / year.
**Total inference cost:** &lt; $50 / year.

That is the cost line that closed the deal with Finance.

---

## 5. Observability and SLOs

| Metric | Target | Alert threshold | Owned by |
| ------ | ------ | --------------- | -------- |
| Nightly batch wall time | &lt; 90 s | &gt; 120 s | Me (on-call) |
| p99 per-OD latency | &lt; 25 ms | &gt; 50 ms | Me |
| 80% PI coverage (rolling 7d) | 78&ndash;82% | &lt; 75% on top-50 family | Me |
| Cost per run | &lt; $0.10 | &gt; $0.50 | Me + Finance |
| LoRA swap failures | 0 | &gt; 0.1% | Me |

The dashboard for these is in [portfolio/](portfolio/) under the
**Inference Optimization** tab.

---

## 6. The Applied Scientist takeaway

Pre-project me would have stopped at "BF16 + batching" because that hits
the SLO and the model accuracy is fine. The reason I went further
(`torch.compile`, LoRA pinning, caching, profiler-driven debugging) is
because I was holding the pager. **Every millisecond I left on the table
is a millisecond closer to a 2 a.m. incident I&apos;ll personally have to
debug.** That changes how you think about optimization &mdash; you stop
optimizing for benchmark numbers and start optimizing for sleeping
through the night.

That is the skill I picked up from this project that I would not have
picked up writing papers.
