# Fine-Tuning TimesFM on AA Pricing Data

> **Sarah:** "Zero-shot is amazing for the long tail. But for trunk routes
> like DFW-LAX we have a decade of data and a 0.6 percentage-point fare-error
> improvement on those routes is worth tens of millions a year. We need to
> fine-tune."
>
> **Marcus:** "Then we need to talk about how the fine-tuned weights ship,
> who owns the audit log, and how we roll back."

This doc walks through the three-stage fine-tuning path AA would use, the
training data layout, the evaluation protocol, and the operational shape of
the resulting model. The two engineers narrate the trade-offs.

---

## 1. The three-stage path

```
   Google "google/timesfm-2.5-200m-pytorch"
                  |
                  v
   STAGE 1: Continued pretraining on AA panel    (Sarah owns)
                  |
                  v
   STAGE 2: XReg head fine-tuning on covariates  (Sarah owns)
                  |
                  v
   STAGE 3: Per-route-family LoRA adapters       (Sarah trains, Marcus deploys)
                  |
                  v
   AA RMS Forecast Service                       (Marcus owns)
```

Each stage has a different *blast radius* if it goes wrong, which is why
they are separated.

---

## 2. The data, in AA terms

### 2.1 Sources

| Source | What it is | Latency | Owner |
| ------ | ---------- | ------- | ----- |
| Sabre PSS extract | Per-OD-pair daily ticket transactions; fare basis, fare class, ticketed amount, refund flag | Daily 03:00 CST | Distribution & Sales |
| AAdvantage RMS warehouse | Daily aggregates: avg fare, load factor, bid-price history, fare-class availability | Daily 04:00 CST | Revenue Management |
| Revenue Integrity feed | Voluntary changes, downgrades, ADM/ACM | Daily 05:00 CST | Revenue Integrity |
| Competitor fare files (ATPCO + Sabre fare-shop) | Snapshots of major US carrier fares for the same OD-pair | Hourly | Pricing |
| Schedule (OAG / internal SSIM) | Published flights & equipment for the next 330 days | Daily | Network Planning |
| Fuel curve | Daily WTI + jet-fuel basis | Daily | Treasury |
| OAG events (holidays, school calendars, sports, conventions) | Curated future calendar | Weekly | Pricing |
| Weather feed (NOAA + private vendor) | Historical actuals + 14-day forecast at hub airports | 6-hourly | Operations |

Sarah&apos;s training panel pulls 5 years from each source, keyed on
`(od_pair, departure_date, snapshot_date)`. The triple key matters: a fare
quoted 60 days out is a different observation from the same fare quoted 7
days out, and the model must learn the time-decay shape.

### 2.2 Aggregation

The unit of training is **(OD-pair, calendar-day)** with a 90-day rolling
context. This is the same shape the example script generates synthetically:

```
target  : avg_fare_usd[od, day]
context : avg_fare_usd[od, day - 90 .. day - 1]
horizon : avg_fare_usd[od, day .. day + 29]
covar   : (8 dynamic + 3 static features, all aligned)
```

5 years &times; 365 days &times; ~3,500 OD-pairs with stable history = ~6.4 M
training windows. After deduplication, leakage-cleanup, and stratified
sampling (over-sampling Q4 and underrepresented routes) the final training
set is ~2.8 M windows, ~50 GB tokenized.

### 2.3 Leakage rules

This is where the ML and Systems sides intersect:

> **Sarah:** "Anything that is only known *after* the departure date is
> banned from the covariate list. That includes actual load factor on the
> day, actual no-show rate, actual fare paid by passengers who booked late."
>
> **Marcus:** "And anything that is only available *retroactively* in our
> warehouse but not in the live feature store at inference time is banned
> too. If I can&apos;t produce it at 02:00 CST on the day of scoring, the
> model can&apos;t train on it."

The intersection: lagged features only. Load factor goes in as
`load_factor_lag_7d`. Competitor fare goes in as the previous day&apos;s
file, not the same-day file. Weather goes in as the published forecast at
T-1, not the actual.

### 2.4 Cohort splits

| Cohort | Time range | Use |
| ------ | ---------- | --- |
| Train | 2019-01 to 2024-09 | Continued pretraining + XReg fit |
| Val | 2024-10 to 2024-12 | Hyperparameter selection, calibration |
| Test (frozen) | 2025-01 to 2025-04 | Final hold-out, never seen during dev |
| Backtest (rolling) | last 30 days | Production drift signal |

COVID-era data (2020-03 to 2021-06) is included with a learned "regime
indicator" static covariate so the model can place that period in a
separate basin. Sarah&apos;s ablation showed that **dropping COVID is worse
than including it with a flag** &mdash; the flag-based approach gives the
model a way to isolate the regime, and even helps with future shock
scenarios because it learned what an "atypical regime" looks like.

---

## 3. Stage 1 &mdash; Continued pretraining

### Goal
Adapt the foundation model&apos;s implicit prior from "all kinds of time
series on the internet" to "AA airline-fare time series".

### Recipe

- Same next-patch loss as the original TimesFM training.
- Learning rate: 1e-5 (10&times; lower than original pretraining; we are
  fine-tuning, not learning from scratch).
- Cosine schedule with 1% warmup.
- 2 epochs over the 2.8 M-window training set; ~14 hours on 8&times;A100.
- Batch size: 256 windows.
- No covariates yet &mdash; only the target series.

### Why no covariates in Stage 1
The transformer trunk is shared across every covariate combination. Pushing
covariates in too early couples the trunk weights to the regression layer
and makes ablation impossible. Sarah keeps the trunk pristine in Stage 1 and
only touches it again in Stage 3 with LoRA.

### Evaluation gate
Continued pretraining is accepted if and only if:

| Metric (on val cohort, no covariates) | Threshold |
| ------------------------------------- | --------- |
| MAPE | &le; 8.5% (vs 10.2% zero-shot baseline) |
| 80% PI coverage | 78&ndash;82% |
| Median forecast bias on Q4 cohort | &le; &plusmn; 1.2% |
| Calibration ECE | &le; 0.04 |

If any threshold fails, fall back to Google&apos;s base weights.

### Marcus&apos;s deployment note

> "The output of Stage 1 is `aa-timesfm-2.5-200m-base-vYYYYMMDD` in our model
> registry. Same shape as the Google checkpoint, same loader, same memory
> footprint. From the platform&apos;s perspective it is a drop-in
> replacement. That&apos;s why we did it this way: the rest of the inference
> stack does not need to change."

---

## 4. Stage 2 &mdash; XReg head fine-tuning

### Goal
Teach the regression layer that consumes covariates the AA-specific
elasticities: how fuel translates to fare on each route family, how holiday
flags map to fare uplifts, how competitor fare moves get matched.

### Recipe

- Freeze the transformer trunk from Stage 1.
- Train only the regression layer that maps `(timesfm_baseline_residual,
  covariate_block)` to the final forecast adjustment.
- L2-regularized linear core with one-hot encoding of categoricals; rank-32
  cross-route interaction term.
- Loss: pinball loss across all 10 quantiles + median MSE, equally weighted.
- 3 epochs; ~40 minutes on 1&times;A10G &mdash; cheap.

### What the layer learns (illustrative)

After training, Sarah inspects the learned coefficients:

| Covariate | Coefficient (avg, $) | Interpretation |
| --------- | -------------------- | -------------- |
| `holiday_flag == 2` (Thanksgiving/Christmas peak) | +$87 | Peak-day premium |
| `holiday_flag == 1` (federal holiday) | +$26 | Federal holiday premium |
| `weather_disruption == 1` | -$19 | Irrops dump |
| `school_break == 1` &times; `route_type == leisure-hub` | +$22 | Spring/summer break premium on leisure |
| `jet_fuel_usd_gal` (per +$1) | +$11.4 | Fuel pass-through |
| `competitor_fare` (per +$10) | +$5.8 | ~58% match elasticity |
| `days_to_departure` (per -1 day) | +$1.3 | Inventory thinning |
| `load_factor_lag` (per +0.1) | +$8.0 | Demand-momentum lift |

These match the team&apos;s domain priors within ~10%, which is the first
sanity check Sarah does before any quantitative evaluation.

### Evaluation gate

| Metric (on val cohort, with covariates) | Threshold |
| ---------------------------------------- | --------- |
| MAPE | &le; 6.0% |
| Q4-only MAPE | &le; 7.5% |
| Holiday-day MAPE | &le; 9.0% |
| 80% PI coverage | 78&ndash;82% |
| Holiday-day 80% PI coverage | 75&ndash;85% |

Stage 2 produces `aa-timesfm-2.5-200m-xreg-vYYYYMMDD`. Same registry, same
loader, but now `forecast_with_covariates` returns AA-tuned predictions
instead of generic ones.

---

## 5. Stage 3 &mdash; Per-route-family LoRA adapters

### Goal
Squeeze out the last 0.5&ndash;1 percentage point of MAPE on the **top 200
OD-pairs by revenue**, where it actually moves the needle.

### Recipe

- Group the 200 OD-pairs into ~20 route families (transcon, transatlantic,
  Caribbean leisure, Mexico leisure, Asia, deep-South, etc.).
- For each family, train **rank-8 LoRA adapters** on the attention
  Q/K/V/O projections of the transformer trunk.
- 1 epoch per family; ~6 minutes each on 1&times;A10G; total ~2 hours for all
  20 families.
- Adapter file: ~4 MB safetensors per family.

### Why LoRA, not full fine-tuning per family

| Option | Disk per family | Train time per family | Inference cost |
| ------ | --------------- | -------------------- | -------------- |
| Full fine-tune | 800 MB | hours, on multi-GPU | re-load 800 MB per family |
| LoRA rank-8 | ~4 MB | ~6 min on 1 GPU | hot-swap in &lt;10 ms |
| Per-route ARIMAX (status quo) | ~50 KB &times; 200 = 10 MB | hours each | 6 ms inference, but stale cold-start |

LoRA is the sweet spot for AA&apos;s scale. Marcus can store all 20 adapters
on the inference node&apos;s local SSD and swap them at batch time.

### Evaluation per family

A family ships only if its LoRA-adapted forecast beats the Stage 2 model on
the same family by at least 0.5 percentage points MAPE on the val cohort and
maintains 80% PI coverage in [78%, 82%].

If a family fails, the inference path falls back to the Stage 2 model with
no adapter loaded. Marcus designed this fallback explicitly: an adapter
failing at 02:00 CST should never block the nightly run.

---

## 6. Inference path with adapters

```
AA Nightly Forecast Service (Marcus)
+----------------------------------+
| 1. Load aa-timesfm-2.5-xreg base |   <- once, ~14 GB on A10G with 64-batch
|    + XReg layer                  |
+----------------------------------+
                |
                v
+----------------------------------+
| 2. For each OD-pair:             |
|    - lookup route_family         |
|    - if family has live adapter: |
|         hot-swap LoRA            |   <- ~10 ms
|         forecast_with_covariates |
|    - else:                       |
|         forecast_with_covariates |   <- Stage 2 model only
|         (no adapter)             |
+----------------------------------+
                |
                v
+----------------------------------+
| 3. Write forecasts to feature    |
|    store, partitioned by         |
|    departure_date                |
+----------------------------------+
                |
                v
       Existing C++ RMS optimizer
       (reads from feature store)
```

### Marcus&apos;s SLOs for this service

| SLO | Target | Alert |
| --- | ------ | ----- |
| Nightly run completion | by 02:30 CST | page on miss |
| 80% PI coverage on top-200 routes (rolling 7d) | 78&ndash;82% | warn @ 75%, page @ 70% |
| Forecast freshness (max staleness) | &lt; 24 h | page @ 36 h |
| Adapter load failure rate | &lt; 0.1% | page @ 1% |
| Cost per nightly run | &lt; $25 | warn @ $40 |

If coverage drops, Sarah is paged. If runtime/cost drift, Marcus is paged.
The split is deliberate: model issues vs system issues route to the right
on-call.

---

## 7. Retraining cadence

| Stage | Cadence | Trigger |
| ----- | ------- | ------- |
| Stage 1 (continued pretraining) | every 6 months | scheduled, plus regime-shift triggers |
| Stage 2 (XReg head) | every 4 weeks | scheduled, plus elasticity drift &gt; 15% |
| Stage 3 (LoRA per family) | every 2 weeks | scheduled, plus per-family MAPE drift |

A single scheduled run never trains all three stages on the same night. The
calendar staggers them so a bad Stage 1 update can&apos;t take down the
whole service &mdash; and so the rollback path (always to the last passing
artifact in each stage) is unambiguous.

---

## 8. Audit, compliance, and the boring stuff that matters

DOT regulations on fare display do not regulate the *model*, they regulate
the *fare displayed to the customer*. But the optimizer that turns the
forecast into a displayed fare is regulated. So:

1. Every forecast write is **immutable**, partitioned by run-id, retained
   for 7 years.
2. Every model artifact is signed with the engineer&apos;s key and the
   training data hash.
3. Every adapter swap is logged with `(od_pair, departure_date, adapter_id,
   forecast_run_id)`.
4. The `xreg + timesfm` decision boundary is explainable per-day: the
   forecast can be decomposed into `timesfm_baseline + sum_of_covariate_contributions`,
   which is what we hand to compliance when they ask "why did the fare go
   up $87 on Thanksgiving?".

The decomposition is exactly what panel (1,1) of the example
visualization shows. That is not a coincidence &mdash; the synthetic example
was deliberately built to mirror what production explainability looks like.

---

## 9. The opening question, revisited

> **Marcus:** "How do the fine-tuned weights ship, who owns the audit log,
> and how do we roll back?"
>
> **Sarah:** "Stage 1 ships as a base-model artifact, Stage 2 ships as the
> XReg head, Stage 3 ships as 20 LoRA files keyed by route family. Audit log
> is per-stage, written to BigQuery, retained 7 years. Rollback is per-stage:
> revert one of (base, head, adapter), the other two stay. The forecast
> service can run with any combination, with adapter being the most
> aggressive lever and base being the most conservative."
>
> **Marcus:** "Good. I&apos;ll wire that into the registry next sprint. We
> deploy Stage 2 to canary on five low-revenue OD-pairs first; if PI
> coverage holds for 7 days we expand to 5%, then 25%, then full."

This doc is the artifact that conversation produced.
