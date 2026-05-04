# American Airlines Pricing Forecast with TimesFM

> *"The fare you see at 9 a.m. on aa.com is not a number. It is the output of a
> revenue-management system answering one question 200 million times a day:
> if I sell this seat for $X, what is the expected value of the next-best
> alternative use of that seat?"*
> &mdash; opening line of the internal RMS onboarding doc that nobody reads,
> paraphrased.

This example adapts the [covariates-forecasting](../covariates-forecasting/)
demo to the airline-pricing problem and walks through it as if two engineers
at American Airlines &mdash; **Sarah, an ML Engineer**, and **Marcus, an ML
Systems Engineer** &mdash; were pairing on it for the upcoming Q4 booking wave.

The folder is self-contained:

| File | Purpose |
| ---- | ------- |
| [forecast_aa_pricing.py](forecast_aa_pricing.py) | Runnable script: synthetic AA panel, XReg API call, 2&times;2 plot |
| [model_architecture.md](model_architecture.md) | How TimesFM works internally and why a foundation model fits pricing |
| [finetuning_aa_data.md](finetuning_aa_data.md) | How AA would fine-tune TimesFM on its own RMS data |
| [scenarios.md](scenarios.md) | Six end-to-end pricing scenarios with the engineers narrating |
| `output/` | Generated plot, panel CSV, metadata JSON (created on first run) |

---

## The cast

**Sarah Chen &mdash; ML Engineer, AA Revenue Management AI**
Background: PhD in operations research, three years at AA. She owns the
forecasting *model* &mdash; data prep, model choice, training loop, evaluation,
calibration. Her north-star metric is *post-departure fare error* and *PI
coverage* on held-out flights.

**Marcus Okafor &mdash; ML Systems Engineer, AA Revenue Management Platform**
Background: distributed systems and feature stores at a hyperscaler before AA.
He owns the *path* the model takes to production &mdash; the feature store,
batch & online inference, drift monitoring, rollback, and the SLOs that keep
the call center, aa.com, the Sabre PSS, and partner channels in sync. His
north-star metrics are *p99 inference latency* and *time-to-rollback*.

They sit on opposite sides of the same line. Sarah ships models that are
correct; Marcus ships a system where the correct model can be the one that
matters at 03:14 a.m. when O&apos;Hare has a ground stop.

---

## The problem, in one paragraph

American Airlines operates ~6,700 flights a day across ~350 destinations.
Each origin-destination (OD) pair has its own demand curve, competitor
landscape, and inventory shape. The Revenue Management System (RMS) needs a
forward-looking estimate of *daily average fare* (and demand) for the next
30 days so it can:

1. Set **bid prices** &mdash; the marginal value of releasing one more seat at
   each fare class.
2. Decide **fare-class availability** &mdash; should we still sell Q-class on
   AA-DFW-LAX next Tuesday, or is the flight tracking too hot?
3. **Match competitors** &mdash; when Delta drops $20 on ATL-LAX at 6:14 a.m.,
   how do we respond on adjacent OD-pairs?
4. **Re-price during irrops** &mdash; a snowstorm at ORD changes the supply
   curve for three days; the inventory must be repriced.

Historically AA used a hand-tuned ARIMAX + neural-net hybrid per route family.
Sarah&apos;s pitch: replace the per-route model with **one foundation model
(TimesFM 2.5)** that handles every OD-pair in one batched call, conditioned
on the covariates the team already publishes &mdash; fuel, competitor fares,
schedule, days-to-departure, holidays, weather.

---

## How the conversation goes

### Day 1, 9:30 a.m. &mdash; the kickoff

> **Sarah:** I want to try TimesFM. It&apos;s a 200-million-parameter
> decoder-only transformer pretrained on 100 billion time-series points.
> It&apos;s zero-shot decent on retail, energy, weather. I think we can use it
> straight off the shelf for the long tail of routes where we don&apos;t have
> enough history to fit a custom model.
>
> **Marcus:** Long tail meaning?
>
> **Sarah:** New OD-pairs, seasonal routes, anything we re-time more than
> twice a year. About 28% of our OD inventory has fewer than 90 days of
> stable history. The current pipeline cold-starts those with a global
> mean-reversion prior, which is why our Q4 fare error blows up on
> Caribbean leisure routes every November.
>
> **Marcus:** OK, but our online RMS is on a 50 ms p99 budget per OD-pair
> repricing call. A 200M-param transformer is not going to do that. So
> what is the actual deployment shape?
>
> **Sarah:** Batch nightly, not online. We score 6,700 OD-pairs &times; 365
> departure-days at 02:00 CST, write 30-day forecasts to the feature store,
> and the existing online RMS reads from there during the day. The
> transformer is the *forecast generator*; the bid-price math stays in the
> existing C++ optimizer.
>
> **Marcus:** That I can build. Forecast as a feature, not as a serving path.
> What&apos;s the input shape?
>
> **Sarah:** Per OD-pair: 90 days of daily average fare, plus eight covariates
> aligned to the same daily grid &mdash; fuel, competitor fare, days-to-departure,
> load-factor lag, holiday flag, day-of-week, school-break flag, weather
> disruption. Plus three static features: route type, cabin, equipment.
>
> **Marcus:** And the output?
>
> **Sarah:** Median fare and ten quantile slices for the next 30 days. The
> RMS optimizer wants the q10 and q90 for risk-aware bid prices, not just
> the point forecast.

That conversation maps directly onto the call in
[forecast_aa_pricing.py](forecast_aa_pricing.py):

```python
point_fc, quant_fc = model.forecast_with_covariates(
    inputs=[fare_dfw_lax, fare_mia_jfk, fare_ord_las],
    dynamic_numerical_covariates={
        "jet_fuel_usd_gal":   [fuel,   fuel,   fuel],
        "competitor_fare":    [comp_a, comp_b, comp_c],
        "days_to_departure":  [dtd_a,  dtd_b,  dtd_c],
        "load_factor_lag":    [lf_a,   lf_b,   lf_c],
    },
    dynamic_categorical_covariates={
        "holiday_flag":       [hol, hol, hol],
        "day_of_week":        [dow, dow, dow],
        "school_break":       [sb,  sb,  sb],
        "weather_disruption": [wx,  wx,  wx],
    },
    static_categorical_covariates={
        "route_type": ["domestic-hub", "leisure-hub", "event-leisure"],
        "cabin":      ["main", "main", "main"],
        "equipment":  ["narrowbody", "narrowbody", "narrowbody"],
    },
    xreg_mode="xreg + timesfm",
)
```

> **Sarah:** &lt;runs the script&gt; First-pass MAPE on the synthetic panel is
> ~6.8%, 80% PI coverage is 79%. That&apos;s already comparable to the
> production hybrid. On real data with 2&times; the context I&apos;d expect
> 4&ndash;5%.
>
> **Marcus:** Coverage of 79% means our PIs are honest. Good. That goes into
> the alerting layer &mdash; if production coverage drops below 70% for any
> route family for three nights running, we page.

---

## The two perspectives, side by side

For every part of this system, Sarah cares about one thing and Marcus cares
about another. The two views are how the example is organized:

| Concern | Sarah (ML Engineer) view | Marcus (ML Systems Engineer) view |
| ------- | ------------------------ | --------------------------------- |
| **Data** | Is the target stationary? Are covariates aligned to the same daily grid? Are holidays in UTC or local-time? | Where does the data live? Sabre PSS extract job, AAdvantage warehouse, Revenue Integrity feed. Latency budget for each. |
| **Model** | Does TimesFM generalize to new OD-pairs? How calibrated are the quantiles? | Model size on disk, RAM at inference, GPU saturation, throughput per batch, cost per forecast. |
| **Covariates** | Which covariates have predictive power? How do I avoid leakage? | How do I guarantee the future schedule of holiday/competitor/fuel covariates is fresh? What&apos;s the staleness SLO? |
| **Output** | MAPE, sMAPE, RMSE, q10/q90 coverage, calibration plots, residual autocorrelation. | Forecast write throughput, downstream RMS read latency, schema versioning, idempotent reruns. |
| **Failure** | Did the model degrade? Re-train. | Did the pipeline degrade? Roll back to yesterday&apos;s forecasts and page on-call. |
| **Iteration** | Notebook + held-out flights. | Shadow deploy, canary 1% of OD-pairs, blue/green RMS feature-store reads. |

The reference materials in the timesfm-forecasting skill align to both views:

- Sarah lives in [api_reference.md](../../references/api_reference.md) and
  [data_preparation.md](../../references/data_preparation.md).
- Marcus lives in [system_requirements.md](../../references/system_requirements.md)
  and [check_system.py](../../scripts/check_system.py).

The very first thing Marcus did before the kickoff above was run

```bash
python timesfm-forecasting/scripts/check_system.py \
  --num-series 6700 \
  --context-length 90 \
  --horizon 30 \
  --batch-size 64 \
  --estimate-only
```

to check that one A10G can hold the whole AA OD universe in a single nightly
sweep. (It can &mdash; ~3 GB GPU memory peak.)

---

## How TimesFM is finetuned to AA pricing data

Out of the box TimesFM is **zero-shot**. AA can use it that way for the long
tail. But for trunk routes (DFW-LAX, MIA-JFK, ORD-LAS, JFK-LHR &hellip;) Sarah
wants to fine-tune so the model picks up AA-specific elasticities &mdash; for
example, that DFW-LAX customers are 30% less price-sensitive than ORD-LAS
customers because DFW-LAX is hub-corporate-dominated.

The full procedure is in [finetuning_aa_data.md](finetuning_aa_data.md). The
short version:

1. **Continued pretraining** on 5 years of AA daily fare history across all
   OD-pairs, using the same next-token loss as the original TimesFM paper but
   with AA&apos;s 50 GB tokenized panel.
2. **XReg head fine-tuning** on the regression layer that consumes the
   covariate stack (fuel, competitor, holiday, &hellip;) so the AA-specific
   elasticities are learned end-to-end.
3. **Per-route-family LoRA adapters** for the top 200 OD-pairs &mdash; rank-8
   adapters that can be swapped at inference time without reloading the base
   model.

Marcus&apos;s contribution is the LoRA adapter registry: every adapter is a
~4 MB safetensors file in S3 keyed by `route_family / training_date`. The
nightly batch loads the base model once, then iterates over OD-pairs hot-
swapping adapters in &lt;10 ms each. That&apos;s the kind of detail that does
not appear in the model paper but determines whether the system works at AA
scale.

---

## The scenarios

[scenarios.md](scenarios.md) walks through six concrete situations, each
narrated by Sarah and Marcus together:

1. **Thanksgiving rush** &mdash; the holiday peak in the horizon is +$95 above
   baseline; show how the holiday covariate is pre-published and how the
   forecast q90 widens to absorb peak-day uncertainty.
2. **Fuel-price spike** &mdash; a Brent rally feeds into the `jet_fuel_usd_gal`
   covariate; the elasticity layer translates it into a domain-specific
   pass-through (~12% on DFW-LAX, ~9% on MIA-JFK).
3. **Competitor undercut** &mdash; Delta drops fares on ATL-LAX overnight; the
   competitor covariate updates at 06:00 CST and the forecast for adjacent
   OD-pairs (DFW-LAX, ORD-LAS) shifts before the open of business.
4. **Weather irrops** &mdash; a winter storm hits ORD; the
   `weather_disruption` covariate is set for the affected days and the
   forecast respects the historical -$22 average drop.
5. **New seasonal route launch** &mdash; AA adds JFK-PVG twice-weekly; with
   only 14 days of history TimesFM zero-shot still produces calibrated
   forecasts because of the foundation-model prior.
6. **Cabin upsell** &mdash; we run the same forecast for premium economy on
   the same OD-pair using the static `cabin` covariate; the model produces
   a wider PI because premium-cabin demand has higher idiosyncratic noise.

For each scenario the doc shows: the covariate inputs, the resulting forecast
shape, what the RMS optimizer does with it, and the SLO Marcus is watching.

---

## Running the example

```bash
# 1. (One time) verify your machine can load the model
python timesfm-forecasting/scripts/check_system.py

# 2. Install the XReg extra if you have not already
pip install timesfm[torch] timesfm[xreg]

# 3. Run the AA pricing example
cd timesfm-forecasting/examples/airline-pricing-forecast
python forecast_aa_pricing.py
```

Expected outputs in `output/`:

| File | Acceptance |
| ---- | ---------- |
| `aa_pricing_covariates.png` | 2&times;2 panel; horizon shaded; legends visible |
| `aa_pricing_panel.csv` | 360 rows (3 routes &times; 120 days) |
| `aa_pricing_metadata.json` | `dimensions.csv_rows` == 360, `xreg_mode` == "xreg + timesfm" |

The script as shipped runs **without** loading TimesFM &mdash; it generates the
synthetic panel and prints the API call you would make. To execute the actual
forecast, paste the printed snippet into a notebook (or into the script after
the `demonstrate_api()` call) once you have `pip install timesfm[xreg]`
installed and ~4 GB of free RAM. Sarah&apos;s notebook for real RMS data
follows the exact same shape but reads from a Sabre extract instead of the
synthetic generator.

---

## What you should have learned by the end

- Why AA pricing is a **multivariate-conditional univariate** problem and how
  TimesFM&apos;s `forecast_with_covariates` API maps onto it.
- How the static / dynamic-numerical / dynamic-categorical covariate
  taxonomy aligns with AA&apos;s actual feature catalog.
- How an ML Engineer and an ML Systems Engineer divide the work and the
  metrics across the same model.
- How TimesFM is fine-tuned on AA data without leaving the foundation-model
  paradigm.
- Six concrete scenarios where the forecast feeds the bid-price optimizer.

If you want the deeper architectural explanation, read
[model_architecture.md](model_architecture.md). If you want the AA-specific
fine-tuning recipe, read [finetuning_aa_data.md](finetuning_aa_data.md). If
you want the operational scenarios, read [scenarios.md](scenarios.md).

---

> *Disclaimer: this example is built on synthetic data designed to mirror
> the shape of real airline pricing signals. It is not connected to any
> actual American Airlines system, contains no proprietary data, and the
> "Sarah" and "Marcus" voices are illustrative composites used to make the
> two-engineer perspective concrete. American Airlines and AAdvantage are
> trademarks of American Airlines, Inc.; their use here is purely
> educational.*
