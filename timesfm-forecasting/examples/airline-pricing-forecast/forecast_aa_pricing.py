#!/usr/bin/env python3
"""
American Airlines Pricing Forecast with TimesFM (XReg Covariates)

Adapted from the retail covariates example to airline ticket pricing.
The target series is daily average fare ($) per route over a 90-day window;
the goal is to forecast the next 30 days so the revenue management team
can place inventory bid-prices and review competitor moves.

Why covariates matter here:
  Airline pricing is driven by *known* future signals -- the calendar of
  holidays is fixed, the schedule is published, fuel hedges are booked,
  competitor fare files are observable, and days-to-departure (DTD) is a
  deterministic clock. TimesFM's XReg API is designed exactly for this case:
  feed in the future values of those covariates and let the model condition
  on them.

Routes modeled (3 distinct demand profiles):
  AA-DFW-LAX  : transcontinental hub leg, business-skewed weekday peaks
  AA-MIA-JFK  : leisure + business mix, holiday-sensitive
  AA-ORD-LAS  : leisure / event-driven, weekend peaks

Covariates:
  Dynamic numerical:
    - jet_fuel_usd_gal     : daily WTI-linked fuel proxy
    - competitor_fare      : Delta / United matched fare for same OD-pair
    - days_to_departure    : average DTD in the booking curve for that day
    - load_factor_lag      : 7-day-lag load factor (% seats sold)
  Dynamic categorical:
    - holiday_flag         : 0=normal, 1=federal holiday, 2=peak (Thx/Xmas)
    - day_of_week          : 0..6
    - school_break         : 0=in session, 1=spring/summer/winter break
    - weather_disruption   : 0=clear, 1=ground delay program at hub
  Static categorical:
    - route_type           : domestic-hub / leisure-hub / event-leisure
    - cabin                : main / premium-economy / business
    - equipment            : narrowbody / widebody

NOTE ON REAL DATA:
  Production AA pricing data lives in the Sabre PSS, the AAdvantage RMS
  warehouse, and the Revenue Integrity feed. Those sources are confidential
  and never committed to a repo. To run this example against real bookings
  you would land an extract in a tempdir:

      import tempfile
      tmp = tempfile.mkdtemp(prefix="aa_pricing_")
      # df = pd.read_parquet(f"{tmp}/od_daily_fares.parquet")

  This file ships only a synthetic dataset that mirrors the *shape* of the
  real signal so the example is reproducible offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXAMPLE_DIR = Path(__file__).parent
OUTPUT_DIR = EXAMPLE_DIR / "output"

CONTEXT_LEN = 90
HORIZON_LEN = 30
TOTAL_LEN = CONTEXT_LEN + HORIZON_LEN  # 120 days

ROUTES = {
    "AA-DFW-LAX": {
        "route_type": "domestic-hub",
        "cabin": "main",
        "equipment": "narrowbody",
        "base_fare": 320.0,
        "weekday_lift": 45.0,
        "leisure_lift": 0.0,
        "fuel_sensitivity": 12.0,
        "competitor_elasticity": 0.55,
    },
    "AA-MIA-JFK": {
        "route_type": "leisure-hub",
        "cabin": "main",
        "equipment": "narrowbody",
        "base_fare": 260.0,
        "weekday_lift": 18.0,
        "leisure_lift": 35.0,
        "fuel_sensitivity": 9.0,
        "competitor_elasticity": 0.70,
    },
    "AA-ORD-LAS": {
        "route_type": "event-leisure",
        "cabin": "main",
        "equipment": "narrowbody",
        "base_fare": 215.0,
        "weekday_lift": -8.0,
        "leisure_lift": 55.0,
        "fuel_sensitivity": 10.0,
        "competitor_elasticity": 0.85,
    },
}


def generate_pricing_data() -> dict:
    """Build synthetic AA daily-fare panels with covariates.

    The structure mirrors the retail example: a `target` (avg fare $),
    a `components` decomposition so we can show what each covariate
    contributes, and a `covariates` block ready to hand to TimesFM XReg.
    """
    rng = np.random.default_rng(7)
    days = np.arange(TOTAL_LEN)

    # Shared market-wide signals -- one realization for the whole panel.
    fuel = 2.85 + 0.25 * np.sin(2 * np.pi * days / 60) + rng.normal(0, 0.06, TOTAL_LEN)
    fuel = fuel.astype(np.float32)

    holiday_calendar = np.zeros(TOTAL_LEN, dtype=np.int32)
    for d in [27, 28, 60, 61]:        # mid-context federal holidays
        if d < TOTAL_LEN:
            holiday_calendar[d] = 1
    for d in [95, 96, 97, 113, 114]:  # Thanksgiving + Christmas peak in horizon
        if d < TOTAL_LEN:
            holiday_calendar[d] = 2

    school_break = np.zeros(TOTAL_LEN, dtype=np.int32)
    school_break[55:70] = 1
    school_break[105:118] = 1

    dow = (np.arange(TOTAL_LEN) % 7).astype(np.int32)

    weather_disruption = np.zeros(TOTAL_LEN, dtype=np.int32)
    weather_disruption[40:42] = 1      # historical irrops in context
    weather_disruption[100:101] = 1    # forecasted winter storm in horizon

    data: dict = {"routes": {}, "covariates": {}, "components": {}}
    fares_by_route: dict[str, np.ndarray] = {}
    competitor_by_route: dict[str, np.ndarray] = {}
    dtd_by_route: dict[str, np.ndarray] = {}
    load_factor_by_route: dict[str, np.ndarray] = {}

    for route_id, cfg in ROUTES.items():
        # Days-to-departure walks down a bookings curve. Real RMS systems
        # use an aggregated DTD per departure-day; here we use the daily
        # average DTD across all open inventory.
        dtd = np.linspace(60, 14, TOTAL_LEN) + rng.normal(0, 1.2, TOTAL_LEN)
        dtd = np.clip(dtd, 7, 90).astype(np.float32)

        # Competitor fare = AA base + spread + correlated noise.
        comp = (
            cfg["base_fare"]
            - 8.0
            + 25 * np.sin(2 * np.pi * days / 14)
            + rng.normal(0, 14, TOTAL_LEN)
        ).astype(np.float32)

        # Lagged load factor (% seats sold 7 days ago) -- proxy for demand.
        lf = 0.62 + 0.08 * np.sin(2 * np.pi * days / 30) + rng.normal(0, 0.04, TOTAL_LEN)
        lf = np.clip(lf, 0.30, 0.98).astype(np.float32)

        # ----- price components ------------------------------------------
        trend = cfg["base_fare"] * (1 + 0.0008 * days)
        seasonality = 18 * np.sin(2 * np.pi * days / 30)
        noise = rng.normal(0, 9, TOTAL_LEN)
        base = (trend + seasonality + noise).astype(np.float32)

        # Weekday vs weekend pattern; sign depends on route mix.
        weekday_mask = (dow >= 1) & (dow <= 4)
        weekend_mask = (dow == 5) | (dow == 6) | (dow == 0)
        weekday_effect = np.where(weekday_mask, cfg["weekday_lift"], 0.0)
        leisure_effect = np.where(weekend_mask, cfg["leisure_lift"], 0.0)
        dow_effect = (weekday_effect + leisure_effect).astype(np.float32)

        # Holidays: 1 = +$30, 2 = +$95 (peak weeks)
        holiday_effect = np.where(
            holiday_calendar == 2, 95.0, np.where(holiday_calendar == 1, 30.0, 0.0)
        ).astype(np.float32)

        # School break adds leisure-route premium only.
        school_effect = (school_break * cfg["leisure_lift"] * 0.4).astype(np.float32)

        # Fuel pass-through
        fuel_effect = (cfg["fuel_sensitivity"] * (fuel - fuel.mean())).astype(np.float32)

        # Competitor matching: AA partially follows competitor moves.
        competitor_effect = (
            cfg["competitor_elasticity"] * (comp - comp.mean())
        ).astype(np.float32)

        # DTD: closer to departure -> higher fare (inventory thinning).
        dtd_effect = (-1.4 * (dtd - dtd.mean())).astype(np.float32)

        # Weather irrops compress fares short-term (dump inventory).
        weather_effect = (-22.0 * weather_disruption).astype(np.float32)

        fare = (
            base
            + dow_effect
            + holiday_effect
            + school_effect
            + fuel_effect
            + competitor_effect
            + dtd_effect
            + weather_effect
        )
        fare = np.maximum(fare, 80.0).astype(np.float32)

        data["routes"][route_id] = {"fare": fare, "config": cfg}
        data["components"][route_id] = {
            "base": base,
            "dow_effect": dow_effect,
            "holiday_effect": holiday_effect,
            "school_effect": school_effect,
            "fuel_effect": fuel_effect,
            "competitor_effect": competitor_effect,
            "dtd_effect": dtd_effect,
            "weather_effect": weather_effect,
        }
        fares_by_route[route_id] = fare
        competitor_by_route[route_id] = comp
        dtd_by_route[route_id] = dtd
        load_factor_by_route[route_id] = lf

    data["covariates"] = {
        "jet_fuel_usd_gal": {rid: fuel for rid in ROUTES},
        "competitor_fare": competitor_by_route,
        "days_to_departure": dtd_by_route,
        "load_factor_lag": load_factor_by_route,
        "holiday_flag": {rid: holiday_calendar for rid in ROUTES},
        "day_of_week": {rid: dow for rid in ROUTES},
        "school_break": {rid: school_break for rid in ROUTES},
        "weather_disruption": {rid: weather_disruption for rid in ROUTES},
        "route_type": {rid: ROUTES[rid]["route_type"] for rid in ROUTES},
        "cabin": {rid: ROUTES[rid]["cabin"] for rid in ROUTES},
        "equipment": {rid: ROUTES[rid]["equipment"] for rid in ROUTES},
    }
    return data


def create_visualization(data: dict) -> None:
    """2x2 figure -- ALL panels share x-axis = days 0..119.

    (0,0) Daily avg fare per route -- context solid, horizon dashed
    (0,1) DFW-LAX: actual vs baseline (no covariates), with event overlays
    (1,0) Competitor fare covariate for all routes (full 120 days)
    (1,1) DFW-LAX covariate effect decomposition (stacked)
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    route_colors = {
        "AA-DFW-LAX": "#0078d2",   # AA blue
        "AA-MIA-JFK": "#c00",       # AA red
        "AA-ORD-LAS": "#1f6f43",    # contrasting green
    }
    days = np.arange(TOTAL_LEN)

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 11), sharex=True,
        gridspec_kw={"hspace": 0.42, "wspace": 0.32},
    )
    fig.suptitle(
        "American Airlines Daily Fare Forecast -- TimesFM XReg with Pricing Covariates\n"
        "Shared x-axis: Day 0-89 = context (observed) | Day 90-119 = forecast horizon",
        fontsize=13, fontweight="bold", y=1.01,
    )

    def add_divider(ax, label_top=True):
        ax.axvline(CONTEXT_LEN - 0.5, color="#9ca3af", lw=1.3, ls="--", alpha=0.8)
        ax.axvspan(
            CONTEXT_LEN - 0.5, TOTAL_LEN - 0.5, alpha=0.06, color="grey", zorder=0
        )
        if label_top:
            ax.text(
                CONTEXT_LEN + 0.6, 1.01, "<- 30-day horizon ->",
                transform=ax.get_xaxis_transform(),
                fontsize=7.5, color="#6b7280", style="italic",
            )

    # -- (0,0): Fares by route ------------------------------------------------
    ax = axes[0, 0]
    for rid, route_data in data["routes"].items():
        fare = route_data["fare"]
        c = route_colors[rid]
        cfg = route_data["config"]
        lbl = f"{rid} ({cfg['route_type']}, base ${cfg['base_fare']:.0f})"
        ax.plot(days[:CONTEXT_LEN], fare[:CONTEXT_LEN], color=c, lw=2, label=lbl)
        ax.plot(days[CONTEXT_LEN:], fare[CONTEXT_LEN:], color=c, lw=1.5, ls="--", alpha=0.6)
    add_divider(ax)
    ax.set_ylabel("Daily Avg Fare ($)", fontsize=10)
    ax.set_title("AA OD-Pair Average Fare", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.22)
    ratio = (
        data["routes"]["AA-DFW-LAX"]["fare"][:CONTEXT_LEN].mean()
        / data["routes"]["AA-ORD-LAS"]["fare"][:CONTEXT_LEN].mean()
    )
    ax.annotate(
        f"DFW-LAX averages {ratio:.2f}x ORD-LAS\n"
        f"(business hub vs leisure-event mix)\n"
        f"-> route_type is a useful static covariate",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4a017", alpha=0.95),
    )

    # -- (0,1): DFW-LAX actual vs baseline -----------------------------------
    ax = axes[0, 1]
    rid = "AA-DFW-LAX"
    comp_dec = data["components"][rid]
    fare = data["routes"][rid]["fare"]
    base = comp_dec["base"]
    holiday = data["covariates"]["holiday_flag"][rid]
    weather = data["covariates"]["weather_disruption"][rid]

    ax.plot(days[:CONTEXT_LEN], base[:CONTEXT_LEN], color="#9ca3af", lw=1.8, ls="--",
            label="Baseline (no covariates)")
    ax.fill_between(
        days[:CONTEXT_LEN], base[:CONTEXT_LEN], fare[:CONTEXT_LEN],
        where=(fare[:CONTEXT_LEN] > base[:CONTEXT_LEN]),
        alpha=0.35, color="#22c55e", label="Covariate uplift",
    )
    ax.fill_between(
        days[:CONTEXT_LEN], fare[:CONTEXT_LEN], base[:CONTEXT_LEN],
        where=(fare[:CONTEXT_LEN] < base[:CONTEXT_LEN]),
        alpha=0.30, color="#ef4444", label="Covariate suppression",
    )
    ax.plot(days[:CONTEXT_LEN], fare[:CONTEXT_LEN], color=route_colors[rid], lw=2,
            label=f"Actual fare ({rid})")

    for d in range(CONTEXT_LEN):
        if holiday[d] == 2:
            ax.axvspan(d - 0.45, d + 0.45, alpha=0.28, color="darkorange", zorder=0)
        elif holiday[d] == 1:
            ax.axvspan(d - 0.45, d + 0.45, alpha=0.18, color="gold", zorder=0)
        if weather[d] == 1:
            ax.axvspan(d - 0.45, d + 0.45, alpha=0.22, color="#5b8def", zorder=0)

    add_divider(ax)
    ax.set_ylabel("Daily Avg Fare ($)", fontsize=10)
    ax.set_title("AA-DFW-LAX -- Actual vs Baseline (No Covariates)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.22)

    holiday_mask = holiday[:CONTEXT_LEN] >= 1
    weather_mask = weather[:CONTEXT_LEN] == 1
    h_lift = (
        (fare[:CONTEXT_LEN][holiday_mask] - base[:CONTEXT_LEN][holiday_mask]).mean()
        if holiday_mask.any() else 0
    )
    w_drop = (
        (fare[:CONTEXT_LEN][weather_mask] - base[:CONTEXT_LEN][weather_mask]).mean()
        if weather_mask.any() else 0
    )
    ax.annotate(
        f"Holiday days: +${h_lift:+.0f} avg vs baseline\n"
        f"Irrops days: ${w_drop:+.0f} avg vs baseline\n"
        f"Future event calendar must be known for XReg",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4a017", alpha=0.95),
    )

    # -- (1,0): Competitor fare covariate ------------------------------------
    ax = axes[1, 0]
    for rid_ in ROUTES:
        ax.plot(days, data["covariates"]["competitor_fare"][rid_],
                color=route_colors[rid_], lw=2, label=rid_, alpha=0.85)
    add_divider(ax, label_top=False)
    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel("Competitor Fare ($)", fontsize=10)
    ax.set_title("Competitor Fare Covariate -- Full 120 Days",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.22)
    ax.annotate(
        "Competitor fare files are observable in real time\n"
        "AA elasticity: matches 55-85% of competitor moves\n"
        "ORD-LAS most reactive (event-leisure, low loyalty)",
        xy=(0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4a017", alpha=0.95),
    )

    # -- (1,1): DFW-LAX covariate effect decomposition -----------------------
    ax = axes[1, 1]
    rid = "AA-DFW-LAX"
    c = data["components"][rid]
    pieces = [
        ("Holiday", c["holiday_effect"], "darkorange"),
        ("Day-of-week", c["dow_effect"], "#22c55e"),
        ("Competitor match", c["competitor_effect"], "steelblue"),
        ("Fuel pass-through", c["fuel_effect"], "#a855f7"),
        ("Days-to-departure", c["dtd_effect"], "#0e7490"),
        ("Weather irrops", c["weather_effect"], "#5b8def"),
    ]
    cum = np.zeros(TOTAL_LEN, dtype=np.float32)
    for name, arr, color in pieces:
        ax.fill_between(days, cum, cum + arr, alpha=0.65, color=color,
                        step="mid", label=f"{name} (max +/-${np.abs(arr).max():.0f})")
        cum = cum + arr
    ax.plot(days, cum, "k-", lw=1.5, alpha=0.75, label="Total covariate effect")
    ax.axhline(0, color="black", lw=0.9, alpha=0.6)
    add_divider(ax, label_top=False)
    ax.set_xlabel("Day", fontsize=10)
    ax.set_ylabel("Effect on fare ($)", fontsize=10)
    ax.set_title("AA-DFW-LAX -- Covariate Effect Decomposition",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.0, loc="upper right", ncol=1)
    ax.grid(True, alpha=0.22, axis="y")
    ax.annotate(
        "Holiday peaks (+$95) and DTD pressure dominate\n"
        "Weather irrops (-$22) are short-lived dips\n"
        "-> Fares = baseline + signed covariate stack",
        xy=(0.97, 0.55), xycoords="axes fraction", ha="right", fontsize=8,
        bbox=dict(boxstyle="round", fc="#fffbe6", ec="#d4a017", alpha=0.95),
    )

    tick_pos = list(range(0, TOTAL_LEN, 10))
    for row in [0, 1]:
        for col in [0, 1]:
            axes[row, col].set_xticks(tick_pos)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "aa_pricing_covariates.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n Saved visualization: {output_path}")


def demonstrate_api() -> None:
    print("\n" + "=" * 72)
    print("  TIMESFM XREG API -- AA PRICING CALL")
    print("=" * 72)
    print("""
import timesfm

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch",
)
model.compile(timesfm.ForecastConfig(
    max_context=90, max_horizon=30,
    normalize_inputs=True,                   # fares vary $80 - $900+
    use_continuous_quantile_head=True,       # need calibrated PIs for RM
    fix_quantile_crossing=True,
    infer_is_positive=True,                  # fares are strictly > 0
))

point_fc, quant_fc = model.forecast_with_covariates(
    inputs=[fare_dfw_lax, fare_mia_jfk, fare_ord_las],
    dynamic_numerical_covariates={
        "jet_fuel_usd_gal":   [fuel,   fuel,   fuel],
        "competitor_fare":    [comp_a, comp_b, comp_c],
        "days_to_departure":  [dtd_a,  dtd_b,  dtd_c],
        "load_factor_lag":    [lf_a,   lf_b,   lf_c],
    },
    dynamic_categorical_covariates={
        "holiday_flag":       [hol,    hol,    hol],
        "day_of_week":        [dow,    dow,    dow],
        "school_break":       [sb,     sb,     sb],
        "weather_disruption": [wx,     wx,     wx],
    },
    static_categorical_covariates={
        "route_type": ["domestic-hub", "leisure-hub", "event-leisure"],
        "cabin":      ["main", "main", "main"],
        "equipment":  ["narrowbody", "narrowbody", "narrowbody"],
    },
    xreg_mode="xreg + timesfm",
    normalize_xreg_target_per_input=True,
)
# point_fc: (3 routes, 30 days)        -- median fare
# quant_fc: (3 routes, 30 days, 10)    -- q10..q90 for revenue management
""")


def explain_xreg_modes() -> None:
    print("\n" + "=" * 72)
    print("  WHY 'xreg + timesfm' FOR PRICING")
    print("=" * 72)
    print("""
TimesFM XReg supports two modes:

  "xreg + timesfm" (DEFAULT, used here)
    1. TimesFM produces a baseline fare forecast from history alone.
    2. A regression on residuals (actual - baseline) fits the covariates.
    3. Final = TimesFM baseline + XReg adjustment.
    Best when: history captures the dominant signal and covariates explain
    the *residual* structure (holidays, irrops, competitor swings).

  "timesfm + xreg"
    1. A regression learns target ~ covariates first.
    2. TimesFM forecasts the residuals from that regression.
    3. Final = XReg prediction + TimesFM residual.
    Best when: covariates explain the main signal (e.g. fuel-driven price).

For AA daily-fare forecasting we use mode #1: TimesFM already learns the
DOW + monthly cycle from 90 days of history; the regression layer's job is
just to pick up the holiday spikes, irrops dips, and competitor matching
that the foundation model can't see in raw target history.
""")


def main() -> None:
    print("=" * 72)
    print("  AMERICAN AIRLINES PRICING FORECAST -- TIMESFM XREG EXAMPLE")
    print("=" * 72)

    print("\n Generating synthetic AA daily-fare panels with covariates...")
    data = generate_pricing_data()

    print(f"   Routes:         {list(data['routes'].keys())}")
    print(f"   Context length: {CONTEXT_LEN} days")
    print(f"   Horizon length: {HORIZON_LEN} days")
    print(f"   Covariates:     {list(data['covariates'].keys())}")

    demonstrate_api()
    explain_xreg_modes()

    print("\n Creating 2x2 visualization (shared x-axis)...")
    create_visualization(data)

    print("\n Saving panel CSV + metadata...")
    OUTPUT_DIR.mkdir(exist_ok=True)

    records = []
    for rid, rdata in data["routes"].items():
        for i in range(TOTAL_LEN):
            cov = data["covariates"]
            comp = data["components"][rid]
            records.append({
                "route_id": rid,
                "day": i,
                "split": "context" if i < CONTEXT_LEN else "horizon",
                "fare_usd": round(float(rdata["fare"][i]), 2),
                "base_fare_usd": round(float(comp["base"][i]), 2),
                "jet_fuel_usd_gal": round(float(cov["jet_fuel_usd_gal"][rid][i]), 4),
                "competitor_fare": round(float(cov["competitor_fare"][rid][i]), 2),
                "days_to_departure": round(float(cov["days_to_departure"][rid][i]), 1),
                "load_factor_lag": round(float(cov["load_factor_lag"][rid][i]), 4),
                "holiday_flag": int(cov["holiday_flag"][rid][i]),
                "day_of_week": int(cov["day_of_week"][rid][i]),
                "school_break": int(cov["school_break"][rid][i]),
                "weather_disruption": int(cov["weather_disruption"][rid][i]),
                "holiday_effect_usd": round(float(comp["holiday_effect"][i]), 2),
                "competitor_effect_usd": round(float(comp["competitor_effect"][i]), 2),
                "fuel_effect_usd": round(float(comp["fuel_effect"][i]), 2),
                "dtd_effect_usd": round(float(comp["dtd_effect"][i]), 2),
                "weather_effect_usd": round(float(comp["weather_effect"][i]), 2),
                "route_type": cov["route_type"][rid],
                "cabin": cov["cabin"][rid],
                "equipment": cov["equipment"][rid],
            })

    df = pd.DataFrame(records)
    csv_path = OUTPUT_DIR / "aa_pricing_panel.csv"
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}  ({len(df)} rows x {len(df.columns)} cols)")

    metadata = {
        "description": (
            "Synthetic American Airlines daily-fare panels with pricing "
            "covariates for TimesFM XReg demo. Mirrors the shape of real "
            "AA RMS feeds (Sabre PSS / AAdvantage warehouse) but contains "
            "no proprietary data."
        ),
        "routes": {
            rid: {
                **rdata["config"],
                "mean_fare_context": round(float(rdata["fare"][:CONTEXT_LEN].mean()), 2),
                "mean_fare_horizon": round(float(rdata["fare"][CONTEXT_LEN:].mean()), 2),
            }
            for rid, rdata in data["routes"].items()
        },
        "dimensions": {
            "context_length": CONTEXT_LEN,
            "horizon_length": HORIZON_LEN,
            "total_length": TOTAL_LEN,
            "num_routes": len(ROUTES),
            "csv_rows": len(df),
        },
        "covariates": {
            "dynamic_numerical": [
                "jet_fuel_usd_gal", "competitor_fare",
                "days_to_departure", "load_factor_lag",
            ],
            "dynamic_categorical": [
                "holiday_flag", "day_of_week",
                "school_break", "weather_disruption",
            ],
            "static_categorical": ["route_type", "cabin", "equipment"],
        },
        "effect_magnitudes": {
            "holiday_peak":  "+$95 per Thanksgiving/Christmas day",
            "holiday_fed":   "+$30 per federal holiday",
            "school_break":  "+0.4 x leisure_lift on leisure routes",
            "weather_irrops": "-$22 per ground-delay day at hub",
            "competitor_match": "0.55 - 0.85 elasticity to competitor delta",
            "dtd_pressure":  "-$1.4 per extra day of inventory available",
        },
        "xreg_mode": "xreg + timesfm",
        "model": "google/timesfm-2.5-200m-pytorch",
    }

    meta_path = OUTPUT_DIR / "aa_pricing_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved: {meta_path}")

    print("\n" + "=" * 72)
    print("  AA PRICING EXAMPLE COMPLETE")
    print("=" * 72)
    print("""
Key points:
  1. forecast_with_covariates() requires TimesFM 2.5 + pip install timesfm[xreg]
  2. Dynamic covariates (fuel, competitor, DTD, holidays) need values for
     BOTH context AND horizon -- pricing teams already publish all of these.
  3. Static covariates (route_type, cabin, equipment) carry the route DNA.
  4. Median + q10/q90 are what RMS feeds into bid-price construction.
  5. See README.md for the full ML / Systems engineer walkthrough.

Output files:
  output/aa_pricing_covariates.png   -- 2x2 visualization
  output/aa_pricing_panel.csv        -- 360-row daily panel (3 routes x 120 days)
  output/aa_pricing_metadata.json    -- effect magnitudes + dimensions
""")


if __name__ == "__main__":
    main()
