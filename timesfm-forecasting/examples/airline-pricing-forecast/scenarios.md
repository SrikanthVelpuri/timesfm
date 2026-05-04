# Six AA Pricing Scenarios with TimesFM XReg

Each scenario below is the kind of situation Sarah and Marcus actually have
to handle on a quarterly basis. For each one we show:

1. **The trigger** &mdash; what happens in the world.
2. **The covariate inputs** &mdash; what flows into TimesFM.
3. **What the forecast looks like** &mdash; expected shape and PI behavior.
4. **What RMS does with it** &mdash; how the bid-price optimizer responds.
5. **Sarah&apos;s view** &mdash; the model concern.
6. **Marcus&apos;s view** &mdash; the system concern.

These all reference the synthetic example in
[forecast_aa_pricing.py](forecast_aa_pricing.py); the same patterns work on
real RMS data with the same call shape.

---

## Scenario 1 &mdash; The Thanksgiving rush

### Trigger
It is October 22. The 30-day RMS forecast horizon now spans the full
Thanksgiving travel window: Nov 26 (Wed before) through Dec 1 (Sun after).
Historical data shows fares on these days run +$95 above baseline on
domestic-hub routes, +$140 on leisure-hub routes.

### Covariate inputs
The pricing team has maintained the federal-holiday calendar for years, so
the `holiday_flag` covariate is already populated for the horizon:

```python
holiday_flag = [
    # ... 90 days context with mostly 0s ...
    0, 0, 0, 0, 0,
    # day 95-97: Thanksgiving peak
    2, 2, 2,
    # ... a few normal days ...
    0, 0, 0, 0, 0, 0, 0, 0,
    # day 113-114: Christmas peak
    2, 2,
    0, 0, 0, 0, 0, 0,
]
```

Combined with `school_break = 1` for the Thanksgiving week and
`days_to_departure` continuing its descent, the model has everything it
needs.

### Forecast shape
On day 95 (Wed before Thanksgiving), the median forecast for AA-DFW-LAX
jumps from ~$370 baseline to ~$465. The q90 jumps to ~$525. The PI is
*wider* on holiday days &mdash; the model has learned that peak-day variance
is structurally higher.

### RMS response
The C++ optimizer reads q10/q90 from the feature store and constructs
risk-aware bid prices. With the wider PI it raises the bid-price for
high-yield fare classes (Y, J) and protects more inventory there, while
holding low fare classes closed for a few extra days. Net: +$11M revenue
across the network for the holiday week, vs the prior pre-TimesFM forecast
that consistently *underestimated* peak-day demand because the per-route
ARIMAX models lacked enough holiday observations.

### Sarah&apos;s view

> "The thing that makes this work is that I trained the XReg head on five
> Thanksgivings of data, so the holiday-flag coefficient is fitted to AA&apos;s
> actual elasticity, not a generic prior. And because I weight the loss with
> pinball across all 10 quantiles, the q90 actually widens on holiday days
> instead of staying flat. That&apos;s the difference between a model that
> looks right and a model that the optimizer can use."

### Marcus&apos;s view

> "From the platform side, this is a stress test. The optimizer reads bid
> prices ~200M times a day, and on Thanksgiving week the read rate doubles
> because of front-loading by call-center agents. The forecast write happens
> once a night, so the read path is the bottleneck &mdash; not the model. We
> over-provision the feature store by 4&times; for the two weeks before
> Thanksgiving. That&apos;s a lesson from 2022 when the cache cold-started
> at 6:30 a.m. CST on Black Friday and we lost 8 minutes of pricing."

---

## Scenario 2 &mdash; The fuel-price spike

### Trigger
Brent crude opens up 9% on a Monday morning. Treasury&apos;s daily fuel
curve update at 04:30 CST shows `jet_fuel_usd_gal` rising from $2.85 to
$3.12 over the next 14 days, then partial mean-reversion.

### Covariate inputs
The `jet_fuel_usd_gal` array for the horizon shifts up. Nothing else changes
in the input.

### Forecast shape
TimesFM&apos;s baseline is unchanged (history is the same), but the XReg
layer sees the fuel-curve covariate move and adds the fuel-pass-through
contribution:

| Route | Fuel sensitivity | Effect on day-15 forecast |
| ----- | ---------------- | ------------------------- |
| AA-DFW-LAX | $12 per $1 fuel | +$3.24 |
| AA-MIA-JFK | $9 per $1 fuel | +$2.43 |
| AA-ORD-LAS | $10 per $1 fuel | +$2.70 |

These are small absolute moves but they apply to all 6,700 OD-pairs
simultaneously. Aggregated across the network it is meaningful revenue.

### RMS response
Bid-prices shift up uniformly across the network. The optimizer holds back
~2% more inventory in the lowest fare classes, recovering the marginal cost
of fuel. No competitive shocks &mdash; everyone&apos;s costs went up, so
matching pressure is symmetric.

### Sarah&apos;s view

> "Fuel is the cleanest covariate to validate against because it is a
> commodity input we can read off TradeStation. The fitted coefficient
> from Stage 2 training is +$11.4 per $1 fuel network-average, which is
> within 5% of the operations-research team&apos;s independent
> bottom-up calculation. When the two numbers match within noise we know
> the model is learning the right elasticity."

### Marcus&apos;s view

> "Fuel updates are an SLO test for our covariate freshness. Treasury
> publishes at 04:30 CST and our nightly forecast runs at 02:00 CST &mdash;
> so by definition the *current* nightly forecast does not see today&apos;s
> fuel update. We added an off-cycle re-score job that fires at 05:00 CST
> on days with &gt;5% fuel moves; it overwrites the affected forecasts in
> the feature store before the call center opens at 06:00 CST. The trigger
> threshold (5%) was set based on the Stage 2 elasticity: anything below 5%
> moves bid-prices by less than $1, which is below the optimizer&apos;s
> rounding."

---

## Scenario 3 &mdash; The competitor undercut

### Trigger
At 06:14 a.m. CST, Delta drops fares on Atlanta-LAX by $20 across all main
cabin fare classes. The Sabre fare-shop ingestion picks it up at 06:18.
Adjacent routes (DFW-LAX, ORD-LAX, MIA-LAX) need to be re-evaluated &mdash;
the customer-substitution effect is real.

### Covariate inputs
The `competitor_fare` covariate for DFW-LAX, ORD-LAX, and MIA-LAX updates
to the new Delta level. Nothing else changes.

### Forecast shape
The XReg layer&apos;s competitor-elasticity coefficient (~0.55 for hub
routes, ~0.85 for leisure routes) translates the $20 competitor drop into:

| Route | Match elasticity | Forecast adjustment |
| ----- | ---------------- | ------------------- |
| AA-DFW-LAX (corp-hub) | 0.55 | -$11.00 |
| AA-MIA-JFK (leisure-hub) | 0.70 | -$14.00 |
| AA-ORD-LAS (event-leisure) | 0.85 | -$17.00 |

The forecast median drops, the PI shifts down, q10 may go below the current
bid-price floor.

### RMS response
On corp-hub routes (DFW-LAX, JFK-LHR), the optimizer ignores the
competitor signal &mdash; AA frequent-flier loyalty buffers the substitution.
On leisure routes (ORD-LAS, BNA-MCO), the optimizer matches Delta within
$3, capturing the bulk of the substitution while preserving margin.

### Sarah&apos;s view

> "The model does *not* automatically match competitor moves &mdash; that
> would be both anticompetitive and incorrect (we have customers who would
> pay AA more for AA reasons). What it does is incorporate the competitor
> signal into the *expected fare* given everything else. The optimizer then
> chooses whether to match. Separating the forecast from the policy is the
> only way to keep the system auditable."

### Marcus&apos;s view

> "Competitor undercuts are why we have an off-cycle re-score path at all.
> The same job that handles fuel re-scores triggers on competitor-fare
> deltas above $8 on routes within the same matching cluster. Latency from
> Delta filing the new fare to AA repricing is 14 minutes p50, 38 minutes
> p99. That used to be 2 hours before we wired the foundation model into
> the off-cycle path."

---

## Scenario 4 &mdash; The weather irrops

### Trigger
A winter storm warning is issued for Chicago O&apos;Hare for two days in the
horizon (day 100 and 101). Operations sets the
`weather_disruption` flag to 1 for any route segment touching ORD on those
days.

### Covariate inputs
The `weather_disruption` covariate flips from 0 to 1 for days 100&ndash;101
on AA-ORD-LAS. Other covariates are unchanged.

### Forecast shape
The XReg coefficient for `weather_disruption == 1` is roughly -$22; the
forecast median for ORD-LAS on those days drops by that amount, and the PI
widens because irrops days have higher fare variance (some passengers are
re-accommodated at the originally-paid fare; some are bumped to revenue
flights at last-seat prices; some are refunded).

### RMS response
The optimizer dumps inventory at lower fare classes for those two days
specifically &mdash; better to sell at $180 than to no-show. Bid-prices
drop on the affected days only; surrounding days are unaffected.

### Sarah&apos;s view

> "Weather is the covariate where I&apos;m most worried about feedback
> loops. If we forecast lower fares on irrops days, we sell more cheap
> seats, which makes irrops days *appear* even cheaper in the next training
> cycle. I countered that by training on **scheduled** weather-disruption
> flags, not realized ones. The model learns the historical effect of
> *expected* irrops, not the realized fare collapse. That keeps the
> elasticity stable across retrains."

### Marcus&apos;s view

> "Weather data has the worst freshness profile of any covariate. NOAA
> updates 6-hourly; our private feed updates every 30 minutes. We pin to
> the most recent published forecast at the moment of the nightly run, and
> we re-score on weather updates that flip the disruption flag. When the
> forecast is wrong &mdash; storm dissipated, no actual irrops &mdash; we
> overwrite the affected day-forecasts with the unflagged version. Customer
> price expectations have already been set by the time we know the storm
> missed, so this becomes a margin question, not a fairness question."

---

## Scenario 5 &mdash; The new seasonal route launch

### Trigger
Network Planning announces JFK-PVG (New York to Shanghai) twice-weekly,
starting in 35 days. This is a brand-new OD-pair with zero historical data.
The Director of RMS asks for a forecast anyway.

### Covariate inputs
There is no fare history. The dynamic covariates (fuel, competitor (China
Eastern, Air China, United), days-to-departure, holiday flag) are all
populated. Static covariates: `route_type=international-asia`,
`cabin=mixed`, `equipment=widebody`.

### Forecast shape
This is where TimesFM **earns its foundation-model credentials**. With zero
context, you can&apos;t actually call TimesFM directly &mdash; the API
requires &gt;= 32 historical points. So Sarah uses a two-stage strategy:

1. **Cold-start synthetic prior.** Take the closest analog OD-pairs
   (JFK-HKG, JFK-NRT, ORD-PVG) and average their normalized fare curves
   into a 90-day "synthetic prior" series.
2. **Run TimesFM on the prior + AA covariates.** The covariate stack
   is real, only the historical context is synthetic.
3. **As real bookings accumulate**, replace the synthetic prior with
   actual JFK-PVG history, day by day, until at day 90 the synthetic
   component is fully aged out.

The PI is wide for the first 30 days and gradually tightens. The q10 floor
gives RMS a conservative bid-price; the q90 ceiling protects against the
launch-period bookings being concentrated in higher fare classes (which
historically happens for new long-haul Asia routes).

### RMS response
Conservative bid-prices in the first 14 days &mdash; protect against
mispricing. Aggressive readjustment as actual booking velocity comes in.

### Sarah&apos;s view

> "Cold-start was the *original* reason I wanted a foundation model. With
> ARIMAX you have nothing for new routes &mdash; the cold-start prior is
> just a hand-tuned average. With TimesFM you have an entire foundation
> model that has seen 100 billion time series, *and* you can explicitly
> condition on covariates that are well-defined for the new route. The
> foundation prior + covariate conditioning gives you a defensible day-1
> forecast that is much closer to the right answer than any hand-tuned
> alternative."

### Marcus&apos;s view

> "The launch path is the cleanest end-to-end test of the system because
> there is no &lsquo;old&rsquo; pipeline to compare against. We track day-1
> forecast vs day-30 forecast vs realized fare for every new route launch,
> and the rolling MAPE on launches has dropped from 16% to 8% since
> TimesFM. Day-1 was always the worst, and now it&apos;s only the
> third-worst day."

---

## Scenario 6 &mdash; The cabin upsell forecast

### Trigger
Strategy team asks: "What would happen if we converted 30 economy seats on
DFW-LAX to premium economy starting next month?". RMS needs a fare forecast
for the *premium economy* cabin specifically, not just main cabin.

### Covariate inputs
Same dynamic covariates as before, but the static `cabin` covariate flips
from `main` to `premium-economy` and the `equipment` covariate stays the
same. The historical context becomes the existing premium-economy fares on
DFW-LAX (which exist; the cabin already runs).

### Forecast shape
The forecast median is ~2.6&times; the main-cabin median, which matches the
historical fare ratio on DFW-LAX. But the PI is much *wider*:

| Cabin | Median forecast | q10 | q90 | PI width |
| ----- | --------------- | --- | --- | -------- |
| Main | $370 | $310 | $445 | $135 |
| Premium economy | $960 | $710 | $1,310 | $600 |

This wider PI is structurally correct: premium-economy demand is more
idiosyncratic (corporate buyers, last-minute upgrades, schedule-driven
upsells), and a model that pretended otherwise would mislead the
optimizer.

### RMS response
The optimizer treats premium economy as a distinct fare-class ladder and
sets bid-prices accordingly. On peak days the wide PI means the q90-driven
upper bid-price is high enough that the optimizer protects more inventory
for last-minute upsells.

### Sarah&apos;s view

> "The cabin covariate is a static categorical, but the way it interacts
> with the dynamic covariates is what matters. Holiday days lift premium
> economy *more in absolute dollars* than main cabin even though the
> *percentage* lift is similar &mdash; that&apos;s a learned interaction
> from the rank-32 cross term in Stage 2. Without that interaction the
> model would systematically underestimate Thanksgiving premium-cabin
> fares by 6&ndash;8%."

### Marcus&apos;s view

> "Multi-cabin forecasts triple the row count in the feature store but use
> the same model invocation. From a system perspective, premium-cabin
> forecasts are just another input to the same nightly run &mdash; not a
> separate pipeline. The savings from not building a dedicated cabin
> pipeline are the reason the cabin-mix simulation can be turned around in
> a day instead of a quarter. That&apos;s strategy&apos;s favorite
> capability."

---

## Putting it together &mdash; the operating rhythm

These six scenarios are not edge cases. In a typical AA week:

- **Scenario 1 (holiday)** is in flight 8 weeks of the year.
- **Scenario 2 (fuel)** moves materially ~10 times a year.
- **Scenario 3 (competitor)** triggers ~20 times a day across the network.
- **Scenario 4 (weather)** triggers ~30 times a year per hub.
- **Scenario 5 (new route)** happens 8&ndash;15 times a year.
- **Scenario 6 (cabin simulation)** is a quarterly strategic ask.

The same `forecast_with_covariates()` call serves all six. That is the point
of the foundation-model approach: one model + one inference path, six
qualitatively different business situations, and the difference between
them is entirely in the covariate inputs and the optimizer&apos;s policy.

> **Sarah:** "The model is the boring part. The interesting part is which
> covariates I trust enough to feed it."
>
> **Marcus:** "And the *really* interesting part is what happens when one
> of those covariate sources misses its 4 a.m. SLA. Which is the next doc I
> want to write."
