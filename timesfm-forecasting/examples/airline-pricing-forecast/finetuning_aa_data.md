# Fine-Tuning TimesFM on AA Pricing Data (No Pretraining)

> **Reality check.** American Airlines is not Google. We do not have a budget for
> continued pretraining of a foundation model from scratch &mdash; the GPU bill
> alone would burn a quarter of the team&apos;s annual cloud spend. So this
> project assumes the **Google base weights are frozen** and we lean entirely on
> the cheap, surgical fine-tuning surfaces TimesFM exposes.

This doc covers the AA fine-tuning recipe, scoped to what an Applied
Scientist (you) can ship in a quarter on commodity GPUs.

The two surfaces we touch:

1. **The XReg regression head** (cheap, fast, retrainable weekly).
2. **Per-route-family LoRA adapters** on the transformer trunk (still cheap,
   trained per-family in minutes).

The Google base model never has its weights overwritten. That keeps the
audit story simple, the rollback path one-liner, and the cloud bill under
the threshold that triggers Finance review.

---

## 1. The two-stage fine-tuning path

```
   google/timesfm-2.5-200m-pytorch  (base weights, frozen forever)
                  |
                  v
   STAGE A: XReg head fine-tuning on AA covariates       (~1 GPU-hr)
                  |
                  v
   STAGE B: Per-route-family LoRA adapters               (~6 GPU-hrs total)
                  |
                  v
   AA RMS Forecast Service  (one base + one head + ~20 LoRA files)
```

### Total compute budget

| Stage | Compute | Wall time | Cloud cost (A10G spot) |
| ----- | ------- | --------- | ---------------------- |
| Stage A &mdash; XReg head | 1 &times; A10G | ~40 min | ~$0.70 |
| Stage B &mdash; LoRA &times; 20 families | 1 &times; A10G | ~6 hr | ~$6.50 |
| **Total quarterly retrain** | 1 &times; A10G | ~7 hr | **~$7.20** |

Compare that to ~$45,000 of A100 hours we&apos;d burn doing continued
pretraining on AA&apos;s 5-year panel. The two-stage approach is **&gt;6,000&times;
cheaper** while capturing &gt;85% of the achievable accuracy gain.

---

## 2. The data, in AA terms

### 2.1 Sources

| Source | What it is | Latency | Owner |
| ------ | ---------- | ------- | ----- |
| Sabre PSS extract | Per-OD-pair daily ticket transactions | Daily 03:00 CST | Distribution & Sales |
| AAdvantage RMS warehouse | Daily aggregates: avg fare, load factor, bid-price history | Daily 04:00 CST | Revenue Management |
| Competitor fare files (ATPCO) | Snapshots of major US carrier fares | Hourly | Pricing |
| Schedule (OAG / SSIM) | Published flights & equipment for next 330 days | Daily | Network Planning |
| Fuel curve | Daily WTI + jet-fuel basis | Daily | Treasury |
| Holiday / event calendar | Curated future calendar | Weekly | Pricing |
| Weather feed (NOAA + private) | 14-day forecast at hub airports | 6-hourly | Operations |

### 2.2 Aggregation

The unit of training is **(OD-pair, calendar-day)** with a 90-day rolling
context &mdash; same shape as the synthetic example.

```
target  : avg_fare_usd[od, day]
context : avg_fare_usd[od, day - 90 .. day - 1]
horizon : avg_fare_usd[od, day .. day + 29]
covar   : (8 dynamic + 3 static features, all aligned)
```

For the **head training (Stage A)** we sample ~250K windows from a 2-year
slice (smaller is fine because the head is a small linear model with few
parameters). For **LoRA training (Stage B)** we use ~50K windows per
route-family, sampled from the most relevant cohorts.

### 2.3 Leakage rules

> Anything only known *after* departure: banned.
> Anything not available in the live feature store at 02:00 CST: banned.
> Lagged-only features pass.

Practically:

- Load factor &rarr; `load_factor_lag_7d`.
- Competitor fare &rarr; previous-day&apos;s file, never same-day.
- Weather &rarr; forecast at T-1, never the realized actual.

### 2.4 Splits

| Cohort | Time range | Use |
| ------ | ---------- | --- |
| Train | last 24 months &minus; 6 months | Stage A + Stage B fits |
| Val | 6 months ago &rarr; 3 months ago | Hyperparam selection, calibration |
| Test (frozen) | last 3 months | Final hold-out |
| Backtest (rolling) | last 30 days | Production drift signal |

COVID-era data is excluded (we don&apos;t pretrain, so we don&apos;t need to
teach the model regime shifts &mdash; the foundation prior already saw plenty
of weird time series).

---

## 3. Stage A &mdash; XReg head fine-tuning

### Goal
Teach the regression layer that consumes covariates the AA-specific
elasticities: how fuel translates to fare on each route family, how
holiday flags map to fare uplifts, how competitor fare moves get matched.

### Recipe

- **Freeze the entire transformer trunk &mdash; we never touch base weights.**
- Train only the regression layer that maps `(timesfm_baseline_residual,
  covariate_block)` to the final forecast adjustment.
- L2-regularized linear core with one-hot encoding of categoricals; rank-32
  cross-route interaction term.
- Loss: pinball loss across all 10 quantiles + median MSE, equally weighted.
- 3 epochs over ~250K windows; ~40 minutes on 1&times;A10G.

### What the layer learns (illustrative)

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

These coefficients match the team&apos;s domain priors within ~10%, which is
the first sanity check before any quantitative evaluation.

### Evaluation gate

| Metric (val cohort, with covariates) | Threshold |
| ------------------------------------- | --------- |
| MAPE | &le; 6.0% |
| Q4-only MAPE | &le; 7.5% |
| Holiday-day MAPE | &le; 9.0% |
| 80% PI coverage | 78&ndash;82% |
| Holiday-day 80% PI coverage | 75&ndash;85% |

Stage A ships as `aa-timesfm-2.5-xreg-vYYYYMMDD` &mdash; just the head
weights as a small (~1.2 MB) safetensors file. The base weights are
unchanged.

---

## 4. Stage B &mdash; Per-route-family LoRA adapters

### Goal
Push the last 0.5&ndash;1 percentage point of MAPE on the **top 200
OD-pairs by revenue**, where the dollar lift is large.

### Recipe

- Group the 200 OD-pairs into ~20 route families (transcon, transatlantic,
  Caribbean leisure, Mexico leisure, etc.).
- For each family, train **rank-8 LoRA adapters** on the attention
  Q/K/V/O projections of the transformer trunk.
- 1 epoch per family over ~50K windows; ~6 minutes each on 1&times;A10G.
- Adapter file: ~4 MB safetensors per family.

### Why LoRA, not full fine-tuning per family

| Option | Disk per family | Train time per family | Inference cost |
| ------ | --------------- | --------------------- | -------------- |
| Full fine-tune | 800 MB | hours, multi-GPU | reload 800 MB |
| LoRA rank-8 | ~4 MB | ~6 min, 1 GPU | hot-swap &lt;10 ms |
| Per-route ARIMAX (status quo) | ~50 KB &times; 200 | hours each | 6 ms but stale |

LoRA is the sweet spot for AA scale. All 20 adapters fit in 80 MB &mdash; the
nightly inference node loads them once at startup and swaps per OD-pair
during scoring.

### Why we get away without continued pretraining

The Google base model already saw 100B time-series points. The job of LoRA
is *not* to teach the trunk what time series look like &mdash; it&apos;s to
nudge attention patterns toward AA-specific cyclic structure on each route
family. Rank-8 has more than enough capacity for that nudge, as confirmed
by the ablation:

| Configuration | Val MAPE | Notes |
| ------------- | -------- | ----- |
| Zero-shot (Google base) | 10.2% | Foundation prior only |
| Stage A (XReg head only) | 6.0% | Covariate-aware, family-agnostic |
| Stage A + LoRA rank-4 | 5.4% | Diminishing returns past this |
| Stage A + LoRA rank-8 | 5.1% | **&larr; chosen** |
| Stage A + LoRA rank-16 | 5.0% | Not worth 2&times; storage |
| Stage A + full fine-tune | 4.7% | Out of budget |

Going from 5.1% to 4.7% would cost ~$45K and a quarter of engineering. Not
defensible.

### Evaluation per family

A family ships only if its LoRA-adapted forecast beats the Stage A model on
the same family by &ge; 0.5 percentage points MAPE on the val cohort and
maintains 80% PI coverage in [78%, 82%].

Failed families fall back to the Stage A model with no adapter loaded.

---

## 5. Inference path with adapters

```
AA Nightly Forecast Service
+----------------------------------+
| 1. Load Google base + Stage A    |   <- once, ~1.5 GB on A10G
|    XReg head                     |
+----------------------------------+
                |
                v
+----------------------------------+
| 2. Pre-load all 20 LoRA          |   <- ~80 MB total in CPU pinned mem
|    adapters into pinned memory   |
+----------------------------------+
                |
                v
+----------------------------------+
| 3. For each batch of 64 OD-pairs:|
|    - group by route_family       |
|    - hot-swap LoRA per family    |   <- ~8 ms amortized
|    - forecast_with_covariates    |
+----------------------------------+
                |
                v
+----------------------------------+
| 4. Write forecasts to feature    |
|    store, partitioned by day     |
+----------------------------------+
                |
                v
        Existing C++ RMS optimizer
```

The full inference-optimization story (batching, BF16, `torch.compile`,
adapter pinning, caching) is in
[inference_optimization.md](inference_optimization.md).

---

## 6. Retraining cadence

| Stage | Cadence | Trigger |
| ----- | ------- | ------- |
| Stage A (XReg head) | every 4 weeks | scheduled, plus elasticity drift &gt; 15% |
| Stage B (LoRA per family) | every 2 weeks | scheduled, plus per-family MAPE drift |

A bad Stage A update never takes down Stage B because they are independent
files in the registry &mdash; rollback Stage A by reverting the head; LoRA
adapters keep running on top of the old head.

---

## 7. Audit & rollback

1. Every forecast write is **immutable**, partitioned by run-id, retained
   7 years.
2. Every artifact (head weights, LoRA file) is signed with the engineer&apos;s
   key and the training data hash.
3. Every adapter swap is logged with `(od_pair, departure_date, adapter_id,
   forecast_run_id)`.
4. The decomposition `forecast = timesfm_baseline + sum(covariate_contribs)`
   is recorded per-day, which is what we hand to compliance when they ask
   "why did the fare go up $87 on Thanksgiving?".

Rollback is trivially per-stage: revert the XReg head OR a LoRA file OR
both. Base weights never change, so they can never be the rollback target.

---

## 8. The headline

The Applied Scientist running this project (you) ships:

- One frozen base model (Google).
- One small AA-tuned head file (~1 MB).
- 20 small AA-tuned LoRA files (~80 MB total).
- Quarterly retrain budget: **~$7**.

That is a defensible MLOps story for any RM director. It is also the kind
of scope an Applied Scientist can own end-to-end &mdash; data, modeling,
evaluation, deployment, and monitoring &mdash; without needing to hire a
parallel ML Engineer team. See [user_stories.md](user_stories.md) for how
that ownership played out across the 16-week project.
