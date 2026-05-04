# TimesFM Architecture for AA Pricing &mdash; Engineering Walkthrough

This is the deeper-than-the-paper read of TimesFM, written so that both Sarah
(ML Engineer) and Marcus (ML Systems Engineer) can answer questions in the
same review. The official paper is *A Decoder-Only Foundation Model for
Time-Series Forecasting* (Das et al., ICML 2024); this doc translates it into
"what does that mean when I&apos;m forecasting DFW-LAX fares?".

---

## 1. The shape of the model

TimesFM 2.5 is a **200-million-parameter decoder-only transformer** trained
to do next-patch prediction on tokenized univariate time series. The
architecture, in numbers AA cares about:

| Property | Value | What it means for pricing |
| -------- | ----- | ------------------------ |
| Parameters | 200 M | Loadable on a single A10G or even CPU; ~1.5 GB resident on CPU, ~1 GB VRAM on GPU. |
| Disk size | ~800 MB safetensors | Fits in the AA model registry without S3 lifecycle headaches. |
| Max input context | 16,384 points | At daily granularity that&apos;s 44+ years &mdash; AA never runs out of context. |
| Max horizon | 256 patches &times; patch_len | At 32-point patches that&apos;s a 256+ step horizon; the 30-day RMS forecast is well inside. |
| Patch length (input) | 32 | Every 32 daily fares get tokenized into one patch. |
| Patch length (output) | 128 | Each output patch covers 128 future days (RMS uses the first 30). |
| Quantile head | 10 quantiles | Returns mean + q10..q90 in one shot; q10/q90 feed the bid-price optimizer. |

The "decoder-only" framing is the same architecture family as GPT, but the
tokens are *patches of real numbers*, not subwords.

---

## 2. The pipeline, top to bottom

```
                                  fares (90 daily $)
                                         |
                                         v
                       +-------------------------------------+
                  (a)  | Patch into 32-point chunks          |
                       | -> 3 patches (90 / 32 round up)     |
                       +-------------------------------------+
                                         |
                                         v
                       +-------------------------------------+
                  (b)  | Per-patch input residual MLP        |
                       | (linear -> SwiGLU -> linear, +skip) |
                       | output: 1280-d patch embedding      |
                       +-------------------------------------+
                                         |
                                         v
                       +-------------------------------------+
                  (c)  | Stacked transformer blocks (x 20)   |
                       | causal self-attention + RoPE        |
                       | feed-forward 4x expansion (SwiGLU)  |
                       +-------------------------------------+
                                         |
                                         v
                       +-------------------------------------+
                  (d)  | Output residual MLP                 |
                       | -> 128-step patch + 10-quantile head|
                       +-------------------------------------+
                                         |
                                         v
                              point + quantile forecast
                                         |
                                         v
                       +-------------------------------------+
                  (e)  | XReg layer (optional)               |
                       | residual regression on covariates   |
                       +-------------------------------------+
                                         |
                                         v
                          final forecast for AA RMS
```

### (a) Patching &mdash; "tokenizing" a time series

Real-valued time series are not discrete tokens. TimesFM solves this with
**input patching**: every 32 consecutive observations become one "token".

For AA, with 90 days of context:

```
patch_0 = fare[0..32)     # roughly the first calendar month of context
patch_1 = fare[32..64)
patch_2 = fare[64..90)    # last partial patch, pad-masked
```

The patch is **mask-aware** &mdash; padding flags ride alongside so the
attention layer never confuses real history with padding. (Bug 4-7 in
production was always this: forgetting to mask trailing partial patches.)

**Why it matters for AA.** Because the patch is 32 points, you never get the
"single-day spike confuses the model" failure mode. A rare $400 spike on
Easter Sunday is averaged into a patch with the surrounding context; the
attention layer then decides whether the spike was structural (it was Easter)
or noise.

### (b) Input residual MLP

Every patch is mapped to a 1280-d vector by a small two-layer MLP with a
residual connection. This is where TimesFM gets its **scale invariance**: the
MLP is trained on patches that have first been per-series **z-normalized**
(mean 0, std 1 across context), so the model never sees the raw $260 fare or
the raw $0.0001 retail margin &mdash; it sees normalized shapes.

That&apos;s what `normalize_inputs=True` in `ForecastConfig` controls. It is
the single most important toggle for AA: fares vary from $59 (Spirit-spec
basic economy) to $11,000 (last-seat Flagship First); without normalization
the same model would not generalize across that scale.

### (c) Transformer trunk

20 blocks of causal self-attention with RoPE positional embedding, SwiGLU
feed-forward, and pre-LayerNorm. Standard 2024-era transformer plumbing. Two
things specifically tuned for time series:

1. **Causal masking is patch-level**, not point-level. Patch *t* attends to
   patches 0&hellip;*t*; this is enough to capture daily, weekly, monthly,
   and quarterly seasonality without the quadratic cost of point-level
   attention.
2. **No frequency token in 2.5.** TimesFM 1.0 / 2.0 required passing a
   `freq` flag (0=high-freq, 1=mid, 2=low) so the model could specialize.
   2.5 dropped this because the in-context patches contain enough signal to
   infer frequency. For AA daily data this means one less knob to set wrong.

### (d) Output head &mdash; point + quantile

A second residual MLP maps the trunk&apos;s final hidden state to a 128-step
forecast, plus 10 quantile slices.

The quantile head is the part Sarah cares most about. The bid-price optimizer
does not want a single fare; it wants a distribution. A median-only forecast
of $312 is useless &mdash; the optimizer needs to know whether the q90 is
$345 (peak-day pressure) or $325 (just noise) to decide whether to release
inventory.

`use_continuous_quantile_head=True` enables a more expressive quantile head
that interpolates smoothly between the trained quantile cutpoints; this
gives noticeably better calibration on long horizons. Default off because it
adds ~5 ms of inference; for batch nightly scoring at AA it is essentially
free, so we always enable it.

`fix_quantile_crossing=True` enforces q10 &le; q20 &le; &hellip; &le; q90.
Without this you occasionally see q40 &gt; q50 in regions of high
uncertainty &mdash; harmless for plotting, fatal for an optimizer that
assumes monotonic quantiles.

### (e) The XReg layer

This is the part that makes TimesFM useful for AA pricing rather than just
"a fancy ARIMA replacement".

XReg is a **regression on residuals** approach. Two modes:

| Mode | Pseudocode | When to use |
| ---- | ---------- | ----------- |
| `xreg + timesfm` (default) | `forecast = TimesFM(history); residual_model.fit(actual - forecast ~ covariates); final = forecast + residual_model.predict(future_covariates)` | When TimesFM already captures the dominant signal and covariates explain residual structure (holidays, irrops, competitor moves). **AA picks this.** |
| `timesfm + xreg` | `residual_model.fit(actual ~ covariates); final = residual_model.predict(future_covariates) + TimesFM(history - residual_model.predict(history))` | When covariates dominate the signal (rare for fares). |

Sarah&apos;s reasoning: 90 days of fare history already includes day-of-week
and month-of-year seasonality, so TimesFM can do that part. But 90 days does
not include Thanksgiving (which is in the horizon, not the context), so the
holiday flag has to be told to TimesFM through the residual layer. Same for
the upcoming weather disruption and any competitor undercut that hasn&apos;t
happened yet.

The internal regression is a regularized linear model with one-hot encoding
for categorical covariates and standardization for numerical covariates.
Static categoricals (route_type, cabin, equipment) become per-series fixed
effects.

---

## 3. Why a foundation model rather than per-route ARIMAX

This is the question the Director of RMS will ask. Sarah&apos;s answer:

| Aspect | Per-route ARIMAX | TimesFM 2.5 zero-shot | TimesFM 2.5 fine-tuned (AA) |
| ------ | ---------------- | --------------------- | --------------------------- |
| **Training time** | hours per route, &times; 6,700 routes &times; weekly retrain = unmanageable | none | days, once, then incremental |
| **Cold-start** | impossible &mdash; need 90+ days of clean history | **good** &mdash; foundation prior fills in | **best** &mdash; AA-specific prior |
| **New OD-pair** | months until calibrated | day-1 usable | day-1 usable, AA-tuned |
| **Cross-route knowledge** | none | implicit (pretraining corpus) | explicit (continued pretraining) |
| **Quantile calibration** | post-hoc residual bootstrap | built-in quantile head | built-in + recalibrated for AA elasticities |
| **Operational footprint** | 6,700 model artifacts in registry | 1 model artifact | 1 base + ~200 LoRA adapters |
| **Drift detection** | per-route; noisy | per-cluster; cleaner | per-cluster + adapter-level health |

Marcus&apos;s addition: "From a Systems standpoint, going from 6,700 model
artifacts to 1 + adapters is the difference between a registry that nobody
maintains and a registry I can actually monitor."

---

## 4. What the model does NOT do

Worth being precise about, because RMS engineers will assume otherwise:

1. **It does not do multivariate forecasting.** TimesFM forecasts one target
   per series; covariates condition it but do not get forecast themselves.
   If AA needs a joint demand+fare forecast, that is two TimesFM calls (or
   a separate VAR step on top).
2. **It does not optimize bid-prices.** It produces the conditional fare
   distribution; the existing C++ optimizer turns that into bid prices.
   That separation is intentional &mdash; the optimizer is regulated, audited,
   and subject to fare-rule compliance. Keeping ML out of the optimizer is a
   feature.
3. **It does not handle hierarchical reconciliation.** If AA wants
   OD-level forecasts that sum coherently to network-level, that is a
   `MinT` or `BUTTOM-UP` reconciliation step on top of TimesFM&apos;s
   per-OD output.
4. **It is not real-time.** 50-100 ms per OD-pair on GPU; not suitable for
   the &lt;10 ms online repricing loop. Use it as a feature provider, not a
   serving model.

---

## 5. Memory, throughput, cost &mdash; Marcus&apos;s table

For AA&apos;s nightly batch of 6,700 OD-pairs &times; 90-day context:

| Metric | CPU (16 cores) | A10G GPU | A100 GPU |
| ------ | -------------- | -------- | -------- |
| Resident memory | 1.7 GB | 1.0 GB VRAM | 1.0 GB VRAM |
| Peak memory (batch=64) | 4.2 GB | 3.1 GB VRAM | 3.4 GB VRAM |
| Throughput (series/s) | ~14 | ~480 | ~2,100 |
| Wall time for 6,700 series | ~8 min | ~14 s | ~3.2 s |
| Cost per nightly run | ~$0.04 (spot CPU) | ~$0.02 (A10G spot) | ~$0.07 (A100 spot) |

The cheapest option is actually the A10G &mdash; the A100&apos;s extra
throughput is wasted on a job that fits in 14 seconds. AA runs nightly on
A10G with a 4&times; safety factor; the same fleet handles the off-cycle
"competitor moved, re-score affected OD-pairs" jobs during the day.

---

## 6. The exact ForecastConfig AA uses

```python
import timesfm

cfg = timesfm.ForecastConfig(
    max_context=90,                          # 1 quarter of daily fares
    max_horizon=30,                          # 30-day RMS window
    normalize_inputs=True,                   # fares vary $80 - $11,000
    per_core_batch_size=64,                  # tuned for A10G
    use_continuous_quantile_head=True,       # better long-horizon calibration
    force_flip_invariance=True,              # f(-x) = -f(x); cheap correctness
    infer_is_positive=True,                  # fares are strictly positive
    fix_quantile_crossing=True,              # required for the optimizer
    return_backcast=False,                   # AA does not need the backcast
)
```

Each toggle ties to a specific failure Sarah has hit in production
backtests:

- `normalize_inputs=False` &rarr; widebody premium-cabin forecasts collapsed
  to "stuck at mean" because the absolute scale dominated the gradient.
- `use_continuous_quantile_head=False` &rarr; quantile bands had visible
  steps at day 14 (when the second output patch starts).
- `infer_is_positive=False` &rarr; q10 went negative on low-fare leisure
  routes; the optimizer treated that as "we will pay you $5 to fly".
- `fix_quantile_crossing=False` &rarr; q40 &gt; q50 once a week on routes
  with extreme variance; optimizer crashed.

The point is: every flag in the config is the scar of a real incident, and
should be on by default for any RMS-grade deployment.

---

## 7. Where to go next

- For the AA-specific training data, evaluation protocol, and LoRA adapter
  scheme, see [finetuning_aa_data.md](finetuning_aa_data.md).
- For the six end-to-end scenarios that show the model doing useful work,
  see [scenarios.md](scenarios.md).
- The TimesFM source is in `src/` of this repo; the patching code lives
  in `src/timesfm/timesfm_2p5/` and the XReg layer in
  `src/timesfm/xreg.py`.

If you have questions about the architecture that this doc does not answer,
the right people to ask in priority order are: the TimesFM paper, the
HuggingFace model card, then the AA RMS team&apos;s
`#rms-foundation-model` channel (internal).
