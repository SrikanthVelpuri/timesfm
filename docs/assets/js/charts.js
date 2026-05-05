/* ===========================================================
   AA Pricing Forecast Portfolio - Chart.js dashboards
   All numbers are illustrative based on the synthetic example.
   =========================================================== */

(function () {
  "use strict";

  // -------- Theme helpers --------
  const COLOR = {
    aaBlue: "#0078d2",
    aaBlueLight: "rgba(0,120,210,0.15)",
    aaRed: "#c8102e",
    aaRedLight: "rgba(200,16,46,0.15)",
    green: "#22c55e",
    greenLight: "rgba(34,197,94,0.15)",
    amber: "#f59e0b",
    amberLight: "rgba(245,158,11,0.18)",
    purple: "#a855f7",
    cyan: "#0e7490",
    grey: "#9ca3af",
    greyLight: "rgba(156,163,175,0.18)",
  };

  Chart.defaults.font.family =
    '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  Chart.defaults.color = "#4b5563";
  Chart.defaults.plugins.legend.labels.boxWidth = 14;
  Chart.defaults.plugins.legend.labels.boxHeight = 14;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.titleFont = { weight: "600" };

  // Track of charts so we can resize on tab activation.
  const charts = {};

  function makeChart(canvasId, config) {
    const el = document.getElementById(canvasId);
    if (!el) return null;
    if (charts[canvasId]) {
      charts[canvasId].destroy();
    }
    charts[canvasId] = new Chart(el, config);
    return charts[canvasId];
  }

  // ============== FINE-TUNING TAB ==============

  function chartMapeProgression() {
    makeChart("chart-mape-progression", {
      type: "bar",
      data: {
        labels: [
          "Zero-shot\n(Google base)",
          "+ XReg head\n(Stage A)",
          "+ LoRA r-4",
          "+ LoRA r-8\n(chosen)",
          "+ LoRA r-16",
          "Full fine-tune\n(rejected)",
        ],
        datasets: [
          {
            label: "Validation MAPE (%)",
            data: [10.2, 6.0, 5.4, 5.1, 5.0, 4.7],
            backgroundColor: [
              COLOR.grey,
              COLOR.aaBlue,
              COLOR.purple + "aa",
              COLOR.purple,
              COLOR.purple + "aa",
              COLOR.aaRed,
            ],
            borderRadius: 4,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: { label: function (ctx) { return "MAPE: " + ctx.parsed.y + "%"; } },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + "%"; } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartCostComparison() {
    makeChart("chart-cost-comparison", {
      type: "bar",
      data: {
        labels: [
          "XReg head\n(Stage A)",
          "LoRA per family\n(Stage B, x20)",
          "Two-stage\nTotal (chosen)",
          "Continued pretraining\n(rejected)",
          "Full fine-tune\n(rejected)",
        ],
        datasets: [
          {
            label: "Quarterly compute cost (USD)",
            data: [0.7, 6.5, 7.2, 45000, 12000],
            backgroundColor: [
              COLOR.aaBlue,
              COLOR.purple,
              COLOR.green,
              COLOR.aaRed,
              COLOR.amber,
            ],
            borderRadius: 4,
            borderSkipped: false,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                const v = ctx.parsed.x;
                if (v < 100) return "$" + v.toFixed(2);
                return "$" + v.toLocaleString();
              },
            },
          },
        },
        scales: {
          x: {
            type: "logarithmic",
            beginAtZero: false,
            min: 0.1,
            grid: { color: "#e5e7eb" },
            ticks: {
              callback: function (v) {
                if (v < 1) return "$" + v;
                if (v < 1000) return "$" + v;
                if (v < 1000000) return "$" + (v / 1000) + "K";
                return "$" + (v / 1000000) + "M";
              },
            },
          },
          y: { grid: { display: false } },
        },
      },
    });
  }

  function chartLoraPerFamily() {
    const families = [
      "Transcon Hub", "Caribbean Leisure", "Mexico Leisure", "Domestic East",
      "Domestic South", "Hawaii", "Latin America", "Transatlantic Premium",
      "Asia Long-haul", "Deep South Niche", "Northeast Shuttle",
      "Pac NW Leisure", "Mountain West", "Florida Hub", "Texas Triangle",
      "Canada", "Caribbean Hub", "Atlantic Niche", "Asia Niche",
      "South America Niche",
    ];
    const headOnly = [5.6, 6.1, 5.8, 5.4, 6.0, 6.4, 5.9, 7.2, 7.5, 6.7,
                      5.2, 6.0, 5.8, 5.5, 5.4, 6.1, 6.3, 7.0, 7.4, 7.1];
    const withLora = [4.2, 4.7, 4.5, 4.0, 4.6, 5.1, 4.5, 6.0, 6.3, 6.5,
                      4.0, 4.7, 4.4, 4.1, 4.0, 4.7, 5.0, 6.9, 7.5, 7.2];

    makeChart("chart-lora-perfamily", {
      type: "bar",
      data: {
        labels: families,
        datasets: [
          {
            label: "Head-only MAPE",
            data: headOnly,
            backgroundColor: COLOR.aaBlueLight,
            borderColor: COLOR.aaBlue,
            borderWidth: 1.5,
            borderRadius: 3,
          },
          {
            label: "With LoRA adapter",
            data: withLora,
            backgroundColor: COLOR.purple,
            borderRadius: 3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              afterBody: function (items) {
                const head = headOnly[items[0].dataIndex];
                const lora = withLora[items[0].dataIndex];
                const delta = head - lora;
                if (delta < 0.5) return "FALLBACK to head-only (delta " + delta.toFixed(1) + " pp)";
                return "Adapter ships (delta " + delta.toFixed(1) + " pp)";
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + "%"; } },
          },
          x: { grid: { display: false }, ticks: { font: { size: 10 } } },
        },
      },
    });
  }

  // ============== INFERENCE TAB ==============

  function chartInferenceLatency() {
    makeChart("chart-inference-latency", {
      type: "bar",
      data: {
        labels: [
          "FP32, batch=1\n(eager)",
          "+ batch=64",
          "+ BF16",
          "+ torch.compile",
          "+ LoRA hot-swap\n(final)",
        ],
        datasets: [
          {
            label: "Median latency per OD-pair (ms)",
            data: [280, 18, 9, 6, 6],
            backgroundColor: [
              COLOR.grey,
              COLOR.cyan,
              COLOR.aaBlue,
              COLOR.purple,
              COLOR.green,
            ],
            borderRadius: 4,
            borderSkipped: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) { return ctx.parsed.y + " ms"; },
              afterBody: function (items) {
                const speedups = ["1.0x baseline", "15.6x speedup", "31.1x speedup", "46.7x speedup", "46.7x median, p99 14ms"];
                return speedups[items[0].dataIndex];
              },
            },
          },
        },
        scales: {
          y: {
            type: "logarithmic",
            min: 1,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + " ms"; } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartThroughputCost() {
    makeChart("chart-throughput-cost", {
      type: "bar",
      data: {
        labels: [
          "FP32 batch=1",
          "batch=64",
          "+ BF16",
          "+ torch.compile",
          "Final (production)",
        ],
        datasets: [
          {
            type: "bar",
            label: "Throughput (OD-pairs / sec)",
            data: [3.6, 56, 110, 165, 165],
            backgroundColor: COLOR.aaBlueLight,
            borderColor: COLOR.aaBlue,
            borderWidth: 2,
            yAxisID: "y",
            borderRadius: 3,
          },
          {
            type: "line",
            label: "Cost per nightly run (USD)",
            data: [0.205, 0.075, 0.038, 0.022, 0.011],
            borderColor: COLOR.aaRed,
            backgroundColor: COLOR.aaRedLight,
            yAxisID: "y1",
            tension: 0.3,
            fill: true,
            pointRadius: 4,
            pointBackgroundColor: COLOR.aaRed,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          y: {
            beginAtZero: true,
            position: "left",
            grid: { color: "#e5e7eb" },
            title: { display: true, text: "OD-pairs / sec" },
          },
          y1: {
            beginAtZero: true,
            position: "right",
            grid: { display: false },
            title: { display: true, text: "Cost (USD)" },
            ticks: { callback: function (v) { return "$" + v.toFixed(3); } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  // ============== SCENARIOS TAB ==============

  function chartScenarioFrequency() {
    makeChart("chart-scenario-frequency", {
      type: "bar",
      data: {
        labels: [
          "Holiday rush",
          "Fuel spike",
          "Competitor undercut",
          "Weather irrops",
          "New route launch",
          "Cabin simulation",
        ],
        datasets: [
          {
            label: "Triggers per year (network-wide)",
            data: [56, 10, 7300, 600, 12, 4],
            backgroundColor: [
              COLOR.amber,
              COLOR.purple,
              COLOR.aaRed,
              COLOR.cyan,
              COLOR.green,
              COLOR.aaBlue,
            ],
            borderRadius: 4,
            borderSkipped: false,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) { return ctx.parsed.x.toLocaleString() + " events / year"; },
            },
          },
        },
        scales: {
          x: {
            type: "logarithmic",
            beginAtZero: false,
            min: 1,
            grid: { color: "#e5e7eb" },
            ticks: {
              callback: function (v) {
                if (v >= 1000) return (v / 1000) + "K";
                return v;
              },
            },
          },
          y: { grid: { display: false } },
        },
      },
    });
  }

  // ============== DASHBOARD TAB ==============

  function chartCoverageDrift() {
    const days = Array.from({ length: 30 }, function (_, i) { return "D-" + (29 - i); });
    // Coverage stays mostly in [78, 82] band with two minor dips
    const coverage = [80.1, 79.5, 80.3, 81.2, 80.7, 80.0, 79.8, 81.0, 79.3, 78.8,
                       77.9, 78.4, 79.1, 80.2, 81.1, 80.6, 80.0, 79.7, 79.5, 78.6,
                       77.4, 78.2, 79.0, 79.8, 80.4, 80.9, 80.2, 79.5, 79.0, 79.3];

    makeChart("chart-coverage-drift", {
      type: "line",
      data: {
        labels: days,
        datasets: [
          {
            label: "80% PI coverage (top-50 routes)",
            data: coverage,
            borderColor: COLOR.aaBlue,
            backgroundColor: COLOR.aaBlueLight,
            tension: 0.3,
            fill: true,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
          {
            label: "Target band upper (82%)",
            data: Array(30).fill(82),
            borderColor: "rgba(34,197,94,0.6)",
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
          },
          {
            label: "Target band lower (78%)",
            data: Array(30).fill(78),
            borderColor: "rgba(34,197,94,0.6)",
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
          },
          {
            label: "Page threshold (75%)",
            data: Array(30).fill(75),
            borderColor: COLOR.aaRed,
            borderWidth: 1.5,
            borderDash: [2, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          y: {
            min: 70,
            max: 90,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + "%"; } },
          },
          x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
        },
        interaction: { mode: "index", intersect: false },
      },
    });
  }

  function chartForecastActual() {
    // Day index 0..119: 0..89 context, 90..119 horizon
    const labels = Array.from({ length: 120 }, function (_, i) { return "D" + i; });

    // Synthetic AA-DFW-LAX fare shape (matches forecast_aa_pricing.py base)
    const actual = [];
    for (let i = 0; i < 90; i++) {
      const trend = 320 * (1 + 0.0008 * i);
      const seas = 18 * Math.sin((2 * Math.PI * i) / 30);
      const dow = (i % 7);
      const dowEff = (dow >= 1 && dow <= 4) ? 45 : 0;
      const noise = (Math.sin(i * 11.3) + Math.cos(i * 7.7)) * 6;
      actual.push(Math.max(80, trend + seas + dowEff + noise));
    }

    // Forecast median for horizon
    const forecast = Array(90).fill(null);
    const lower = Array(90).fill(null);
    const upper = Array(90).fill(null);
    for (let i = 90; i < 120; i++) {
      const trend = 320 * (1 + 0.0008 * i);
      const seas = 18 * Math.sin((2 * Math.PI * i) / 30);
      const dow = (i % 7);
      const dowEff = (dow >= 1 && dow <= 4) ? 45 : 0;
      // Holiday peaks in the horizon at days 95-97 (Thanksgiving) and 113-114 (Christmas)
      let holidayEff = 0;
      if ((i >= 95 && i <= 97) || (i >= 113 && i <= 114)) holidayEff = 95;
      const median = trend + seas + dowEff + holidayEff;
      forecast.push(median);
      const piWidth = holidayEff > 0 ? 60 : 30;
      lower.push(median - piWidth);
      upper.push(median + piWidth);
    }

    makeChart("chart-forecast-actual", {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Actual fare (context)",
            data: actual.concat(Array(30).fill(null)),
            borderColor: COLOR.aaBlue,
            backgroundColor: "transparent",
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 0,
          },
          {
            label: "Forecast median (horizon)",
            data: forecast,
            borderColor: COLOR.aaRed,
            borderWidth: 2.2,
            borderDash: [6, 4],
            tension: 0.25,
            pointRadius: 0,
          },
          {
            label: "q10",
            data: lower,
            borderColor: "rgba(200,16,46,0.4)",
            backgroundColor: COLOR.aaRedLight,
            borderWidth: 1,
            tension: 0.25,
            pointRadius: 0,
            fill: "+1",
          },
          {
            label: "q90",
            data: upper,
            borderColor: "rgba(200,16,46,0.4)",
            backgroundColor: "transparent",
            borderWidth: 1,
            tension: 0.25,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { mode: "index", intersect: false },
        },
        scales: {
          y: {
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return "$" + v; } },
            title: { display: true, text: "Daily avg fare (USD)" },
          },
          x: {
            grid: { display: false },
            ticks: { maxTicksLimit: 12 },
          },
        },
      },
    });
  }

  function chartPerRouteMape() {
    const families = ["Transcon", "Caribbean", "Mexico", "Florida", "Hawaii", "Latin Am", "Transatlantic", "Asia"];
    const zeroShot = [10.4, 11.2, 10.0, 9.5, 12.8, 11.0, 14.5, 15.1];
    const headOnly = [5.6, 6.1, 5.8, 5.4, 6.4, 5.9, 7.2, 7.5];
    const withLora = [4.2, 4.7, 4.5, 4.1, 5.1, 4.5, 6.0, 6.3];

    makeChart("chart-per-route-mape", {
      type: "bar",
      data: {
        labels: families,
        datasets: [
          { label: "Zero-shot",   data: zeroShot, backgroundColor: COLOR.greyLight, borderColor: COLOR.grey, borderWidth: 1.5, borderRadius: 3 },
          { label: "+ XReg head", data: headOnly, backgroundColor: COLOR.aaBlueLight, borderColor: COLOR.aaBlue, borderWidth: 1.5, borderRadius: 3 },
          { label: "+ LoRA",      data: withLora, backgroundColor: COLOR.purple, borderRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + "%"; } },
            title: { display: true, text: "Validation MAPE" },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  // ============== MARKET-ROUTES PAGE ==============

  function chartRegionMape() {
    makeChart("chart-region-mape", {
      type: "bar",
      data: {
        labels: ["Domestic (76 routes)", "Intl-short (31 routes)", "Intl-long (21 routes)"],
        datasets: [
          { label: "Zero-shot",   data: [9.8, 11.1, 14.9], backgroundColor: COLOR.greyLight, borderColor: COLOR.grey, borderWidth: 1.5, borderRadius: 3 },
          { label: "+ XReg head", data: [5.5, 6.0, 7.3],   backgroundColor: COLOR.aaBlueLight, borderColor: COLOR.aaBlue, borderWidth: 1.5, borderRadius: 3 },
          { label: "+ LoRA",      data: [4.2, 4.6, 6.1],   backgroundColor: COLOR.purple, borderRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          y: {
            beginAtZero: true,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + "%"; } },
            title: { display: true, text: "MAPE (%)" },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartRegionRevenue() {
    makeChart("chart-region-revenue", {
      type: "bar",
      data: {
        labels: ["Domestic", "Intl-short", "Intl-long"],
        datasets: [
          {
            type: "bar",
            label: "Annual revenue lift (USD M)",
            data: [28.4, 9.6, 10.0],
            backgroundColor: [COLOR.aaBlue, COLOR.green, COLOR.amber],
            borderRadius: 4,
            yAxisID: "y",
          },
          {
            type: "line",
            label: "Lift % of base revenue",
            data: [0.37, 0.40, 0.28],
            borderColor: COLOR.aaRed,
            backgroundColor: "transparent",
            yAxisID: "y1",
            tension: 0.2,
            pointRadius: 5,
            pointBackgroundColor: COLOR.aaRed,
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.datasetIndex === 0) return "$" + ctx.parsed.y + "M annual lift";
                return ctx.parsed.y + "% of region baseline";
              },
            },
          },
        },
        scales: {
          y: {
            beginAtZero: true,
            position: "left",
            grid: { color: "#e5e7eb" },
            title: { display: true, text: "Revenue lift (USD M)" },
            ticks: { callback: function (v) { return "$" + v + "M"; } },
          },
          y1: {
            beginAtZero: true,
            position: "right",
            grid: { display: false },
            title: { display: true, text: "Lift %" },
            ticks: { callback: function (v) { return v + "%"; } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartRouteMapeScatter() {
    // Build scatter from embedded route data (mirrors metrics.json subset).
    const ROUTES = [
      // Domestic (orange) sample
      {"od":"DFW-LAX","z":10.6,"l":4.2,"r":"D"},{"od":"DFW-JFK","z":10.4,"l":4.1,"r":"D"},
      {"od":"DFW-SFO","z":10.3,"l":4.0,"r":"D"},{"od":"DFW-ORD","z":9.8,"l":4.0,"r":"D"},
      {"od":"DFW-MIA","z":9.6,"l":4.1,"r":"D"},{"od":"ORD-LAX","z":10.4,"l":4.1,"r":"D"},
      {"od":"ORD-JFK","z":10.5,"l":4.2,"r":"D"},{"od":"ORD-MIA","z":9.7,"l":4.2,"r":"D"},
      {"od":"JFK-LAX","z":10.7,"l":4.3,"r":"D"},{"od":"JFK-MIA","z":9.5,"l":4.1,"r":"D"},
      {"od":"LAX-SFO","z":9.4,"l":4.4,"r":"D"},{"od":"LAX-LAS","z":9.7,"l":4.5,"r":"D"},
      {"od":"LAX-HNL","z":12.6,"l":5.2,"r":"D"},{"od":"LAX-OGG","z":12.8,"l":5.3,"r":"D"},
      {"od":"BOS-LGA","z":8.7,"l":3.7,"r":"D"},{"od":"BOS-DCA","z":8.9,"l":3.9,"r":"D"},
      {"od":"ORD-LGA","z":9.0,"l":3.9,"r":"D"},{"od":"ORD-BOS","z":9.0,"l":4.0,"r":"D"},
      {"od":"PHX-LAS","z":10.1,"l":4.5,"r":"D"},{"od":"DEN-DFW","z":10.0,"l":4.3,"r":"D"},
      {"od":"ATL-LAX","z":10.4,"l":4.2,"r":"D"},{"od":"CLT-LGA","z":9.4,"l":3.9,"r":"D"},
      {"od":"AUS-DFW","z":9.4,"l":4.0,"r":"D"},{"od":"DFW-PHX","z":10.1,"l":4.4,"r":"D"},
      {"od":"DFW-DEN","z":10.0,"l":4.3,"r":"D"},
      // Intl-short (green)
      {"od":"DFW-CUN","z":11.0,"l":4.5,"r":"S"},{"od":"DFW-MEX","z":10.8,"l":4.4,"r":"S"},
      {"od":"DFW-SJD","z":11.1,"l":4.5,"r":"S"},{"od":"LAX-CUN","z":11.0,"l":4.5,"r":"S"},
      {"od":"LAX-MEX","z":10.7,"l":4.4,"r":"S"},{"od":"ORD-CUN","z":11.1,"l":4.6,"r":"S"},
      {"od":"MIA-NAS","z":11.2,"l":4.7,"r":"S"},{"od":"MIA-SJU","z":11.0,"l":4.6,"r":"S"},
      {"od":"MIA-PUJ","z":11.3,"l":4.7,"r":"S"},{"od":"MIA-MBJ","z":11.4,"l":4.8,"r":"S"},
      {"od":"JFK-SJU","z":11.0,"l":4.6,"r":"S"},{"od":"YYZ-LGA","z":10.4,"l":4.4,"r":"S"},
      {"od":"YVR-LAX","z":10.3,"l":4.3,"r":"S"},
      // Intl-long (red)
      {"od":"JFK-LHR","z":14.5,"l":6.0,"r":"L"},{"od":"JFK-CDG","z":14.4,"l":5.9,"r":"L"},
      {"od":"JFK-FCO","z":14.6,"l":6.1,"r":"L"},{"od":"DFW-LHR","z":14.6,"l":6.1,"r":"L"},
      {"od":"DFW-CDG","z":14.5,"l":6.0,"r":"L"},{"od":"DFW-FRA","z":14.7,"l":6.1,"r":"L"},
      {"od":"ORD-LHR","z":14.5,"l":6.0,"r":"L"},{"od":"ORD-FRA","z":14.6,"l":6.1,"r":"L"},
      {"od":"DFW-NRT","z":15.1,"l":6.3,"r":"L"},{"od":"DFW-PVG","z":15.3,"l":6.4,"r":"L"},
      {"od":"DFW-ICN","z":15.2,"l":6.3,"r":"L"},{"od":"DFW-HKG","z":15.4,"l":6.4,"r":"L"},
      {"od":"LAX-NRT","z":15.0,"l":6.2,"r":"L"},{"od":"LAX-PVG","z":15.2,"l":6.3,"r":"L"},
      {"od":"LAX-SYD","z":15.5,"l":6.5,"r":"L"},{"od":"JFK-DEL","z":15.3,"l":6.4,"r":"L"},
      {"od":"MIA-GRU","z":11.4,"l":4.5,"r":"L"},{"od":"MIA-EZE","z":11.5,"l":4.6,"r":"L"},
    ];
    const dom = ROUTES.filter(function (r) { return r.r === "D"; });
    const sho = ROUTES.filter(function (r) { return r.r === "S"; });
    const lng = ROUTES.filter(function (r) { return r.r === "L"; });

    makeChart("chart-route-mape-scatter", {
      type: "scatter",
      data: {
        datasets: [
          {
            label: "Domestic",
            data: dom.map(function (r) { return { x: r.z, y: r.l, label: r.od }; }),
            backgroundColor: COLOR.aaBlue,
            pointRadius: 5,
          },
          {
            label: "International short-haul",
            data: sho.map(function (r) { return { x: r.z, y: r.l, label: r.od }; }),
            backgroundColor: COLOR.green,
            pointRadius: 5,
          },
          {
            label: "International long-haul",
            data: lng.map(function (r) { return { x: r.z, y: r.l, label: r.od }; }),
            backgroundColor: COLOR.aaRed,
            pointRadius: 5,
          },
          {
            label: "y = x (no improvement)",
            type: "line",
            data: [{ x: 5, y: 5 }, { x: 16, y: 16 }],
            borderColor: COLOR.grey,
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                if (ctx.dataset.type === "line") return "no-improvement diagonal";
                const d = ctx.raw;
                return d.label + ": zero-shot " + d.x + "% → LoRA " + d.y + "%";
              },
            },
          },
        },
        scales: {
          x: {
            title: { display: true, text: "Zero-shot MAPE (%)" },
            grid: { color: "#e5e7eb" },
            min: 5,
            max: 16,
            ticks: { callback: function (v) { return v + "%"; } },
          },
          y: {
            title: { display: true, text: "Post-LoRA MAPE (%)" },
            grid: { color: "#e5e7eb" },
            min: 3,
            max: 8,
            ticks: { callback: function (v) { return v + "%"; } },
          },
        },
      },
    });
  }

  function chartRegionCoverage() {
    const days = Array.from({ length: 30 }, function (_, i) { return "D-" + (29 - i); });
    function jitter(seed, base, amp) {
      const out = [];
      for (let i = 0; i < 30; i++) {
        out.push(base + amp * Math.sin(i * 0.7 + seed) + amp * 0.4 * Math.cos(i * 1.3 + seed));
      }
      return out;
    }
    makeChart("chart-region-coverage", {
      type: "line",
      data: {
        labels: days,
        datasets: [
          { label: "Domestic", data: jitter(1.1, 80.3, 0.8), borderColor: COLOR.aaBlue, backgroundColor: "transparent", borderWidth: 2, tension: 0.3, pointRadius: 0 },
          { label: "Intl-short", data: jitter(2.3, 79.5, 0.7), borderColor: COLOR.green, backgroundColor: "transparent", borderWidth: 2, tension: 0.3, pointRadius: 0 },
          { label: "Intl-long", data: jitter(3.7, 78.2, 0.9), borderColor: COLOR.aaRed, backgroundColor: "transparent", borderWidth: 2, tension: 0.3, pointRadius: 0 },
          { label: "Target band upper (82%)", data: Array(30).fill(82), borderColor: "rgba(34,197,94,0.5)", borderWidth: 1.2, borderDash: [4, 4], pointRadius: 0, fill: false },
          { label: "Target band lower (78%)", data: Array(30).fill(78), borderColor: "rgba(34,197,94,0.5)", borderWidth: 1.2, borderDash: [4, 4], pointRadius: 0, fill: false },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          y: {
            min: 75,
            max: 84,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return v + "%"; } },
            title: { display: true, text: "80% PI coverage" },
          },
          x: { grid: { display: false }, ticks: { maxTicksLimit: 10 } },
        },
        interaction: { mode: "index", intersect: false },
      },
    });
  }

  // Populate the Top-20 routes table on market-routes.html from metrics.json.
  function populateTopRoutesTable() {
    const tbody = document.querySelector("#top-routes-table tbody");
    if (!tbody) return;
    fetch("../data/metrics.json")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        const sorted = (data.routes || []).slice().sort(function (a, b) { return b.rev_lift_m - a.rev_lift_m; });
        const top = sorted.slice(0, 20);
        const rows = top.map(function (r) {
          const region = r.region === "D" ? "Domestic" : "Intl";
          return "<tr>" +
            "<td><strong>" + r.od + "</strong></td>" +
            "<td>" + r.family + "</td>" +
            "<td>" + region + "</td>" +
            "<td>" + r.mape_zs.toFixed(1) + "%</td>" +
            "<td><strong>" + r.mape_lora.toFixed(1) + "%</strong></td>" +
            "<td>" + r.cov.toFixed(1) + "%</td>" +
            "<td>$" + r.fare + "</td>" +
            "<td>$" + r.rev_base_m.toFixed(0) + "M</td>" +
            "<td><strong>$" + r.rev_lift_m.toFixed(2) + "M</strong></td>" +
            "</tr>";
        }).join("");
        tbody.innerHTML = rows;
      })
      .catch(function () {
        tbody.innerHTML = "<tr><td colspan='9'>Could not load route data.</td></tr>";
      });
  }

  // ============== BUSINESS-IMPACT PAGE ==============

  function chartBiRevenueRegion() {
    makeChart("chart-bi-revenue-region", {
      type: "bar",
      data: {
        labels: ["Domestic", "Intl-short", "Intl-long"],
        datasets: [
          { label: "Baseline annual revenue (USD B)", data: [7.6, 2.4, 3.6], backgroundColor: COLOR.aaBlueLight, borderColor: COLOR.aaBlue, borderWidth: 1.5, borderRadius: 3, yAxisID: "y" },
          { label: "Annual revenue lift (USD M)",     data: [28.4, 9.6, 10.0], backgroundColor: COLOR.aaRed, borderRadius: 3, yAxisID: "y1" },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: "top" } },
        scales: {
          y: {
            beginAtZero: true, position: "left", grid: { color: "#e5e7eb" },
            title: { display: true, text: "Baseline (USD B)" },
            ticks: { callback: function (v) { return "$" + v + "B"; } },
          },
          y1: {
            beginAtZero: true, position: "right", grid: { display: false },
            title: { display: true, text: "Lift (USD M)" },
            ticks: { callback: function (v) { return "$" + v + "M"; } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartBiMlByRegion() {
    makeChart("chart-bi-ml-by-region", {
      type: "bar",
      data: {
        labels: ["MAPE", "RMSE", "Pinball q90", "CRPS", "ECE (calibration)"],
        datasets: [
          { label: "Domestic",   data: [57.1, 38.5, 31.2, 33.6, 42.0], backgroundColor: COLOR.aaBlue, borderRadius: 3 },
          { label: "Intl-short", data: [58.6, 41.0, 33.5, 35.4, 39.8], backgroundColor: COLOR.green,  borderRadius: 3 },
          { label: "Intl-long",  data: [59.1, 35.8, 27.4, 29.1, 36.2], backgroundColor: COLOR.aaRed,  borderRadius: 3 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { callbacks: { label: function (ctx) { return ctx.dataset.label + ": -" + ctx.parsed.y + "%"; } } },
        },
        scales: {
          y: {
            beginAtZero: true, grid: { color: "#e5e7eb" },
            title: { display: true, text: "Improvement vs zero-shot (%)" },
            ticks: { callback: function (v) { return v + "%"; } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartBiRolloutTrajectory() {
    const weeks = Array.from({ length: 16 }, function (_, i) { return "W" + (i + 1); });
    // Synthetic cumulative annualised lift trajectory
    const lift = [0, 0, 0, 0, 0, 0.4, 1.6, 4.8, 12.0, 18.4, 24.1, 28.6, 36.3, 41.8, 45.6, 48.0];
    makeChart("chart-bi-rollout-trajectory", {
      type: "line",
      data: {
        labels: weeks,
        datasets: [
          {
            label: "Cumulative annualised lift (USD M)",
            data: lift,
            borderColor: COLOR.aaBlue,
            backgroundColor: COLOR.aaBlueLight,
            tension: 0.3,
            fill: true,
            pointRadius: 4,
            pointBackgroundColor: COLOR.aaBlue,
            borderWidth: 2.5,
          },
          {
            label: "Floor target ($30M)",
            data: Array(16).fill(30),
            borderColor: COLOR.green,
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { callbacks: { label: function (ctx) { return "$" + ctx.parsed.y.toFixed(1) + "M"; } } },
        },
        scales: {
          y: {
            beginAtZero: true, grid: { color: "#e5e7eb" },
            title: { display: true, text: "Cumulative annualised lift (USD M)" },
            ticks: { callback: function (v) { return "$" + v + "M"; } },
          },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function chartBiFamilyRevenue() {
    const families = [
      "Transcon Hub", "Transatlantic Premium", "Asia Long-haul", "Florida Hub",
      "Mexico Leisure", "Caribbean Hub", "Domestic East", "Mountain West",
      "Northeast Shuttle", "Pac NW Leisure", "Caribbean Leisure", "Latin America",
      "Hawaii", "Texas Triangle", "Domestic South", "Canada", "Atlantic Niche",
      "Asia Niche", "South America Niche", "Deep South Niche",
    ];
    const lift = [11.4, 9.8, 6.8, 4.2, 4.4, 3.6, 2.8, 2.6, 2.4, 2.0, 1.8, 1.6, 1.4, 0.9, 0.7, 0.5, 0.2, 0.15, 0.12, 0.10];
    const ships = [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, false, false, false, false];
    makeChart("chart-bi-family-revenue", {
      type: "bar",
      data: {
        labels: families,
        datasets: [
          {
            label: "Annual revenue lift (USD M)",
            data: lift,
            backgroundColor: ships.map(function (s) { return s ? COLOR.aaBlue : COLOR.grey; }),
            borderRadius: 3,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                return "$" + ctx.parsed.x.toFixed(2) + "M" + (ships[ctx.dataIndex] ? "" : " (head-only fallback)");
              },
            },
          },
        },
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: "#e5e7eb" },
            ticks: { callback: function (v) { return "$" + v + "M"; } },
          },
          y: { grid: { display: false }, ticks: { font: { size: 11 } } },
        },
      },
    });
  }

  // ============== INDEX MARKETS TAB ==============

  function chartMarketsRevenueShare() {
    makeChart("chart-markets-revenue-share", {
      type: "doughnut",
      data: {
        labels: ["Domestic ($28.4M)", "Intl-short ($9.6M)", "Intl-long ($10.0M)"],
        datasets: [
          {
            data: [28.4, 9.6, 10.0],
            backgroundColor: [COLOR.aaBlue, COLOR.green, COLOR.aaRed],
            borderWidth: 2,
            borderColor: "#ffffff",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "right" },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                const total = 48;
                const pct = ((ctx.parsed / total) * 100).toFixed(1);
                return ctx.label + " — " + pct + "% of total lift";
              },
            },
          },
        },
        cutout: "60%",
      },
    });
  }

  function chartMarketsRouteCount() {
    makeChart("chart-markets-route-count", {
      type: "bar",
      data: {
        labels: [
          "Transcon Hub", "Florida Hub", "Mountain West", "Northeast Shuttle",
          "Pac NW Leisure", "Texas Triangle", "Domestic East", "Domestic South", "Hawaii",
          "Mexico Leisure", "Caribbean Hub", "Caribbean Leisure", "Canada",
          "Latin America", "Transatlantic Premium", "Asia Long-haul",
        ],
        datasets: [
          {
            label: "OD-pairs fine-tuned",
            data: [16, 9, 8, 8, 6, 5, 5, 4, 3, 12, 5, 7, 4, 10, 16, 14],
            backgroundColor: function (ctx) {
              const dom = ["Transcon Hub", "Florida Hub", "Mountain West", "Northeast Shuttle", "Pac NW Leisure", "Texas Triangle", "Domestic East", "Domestic South", "Hawaii"];
              const sho = ["Mexico Leisure", "Caribbean Hub", "Caribbean Leisure", "Canada"];
              const fam = ctx.chart.data.labels[ctx.dataIndex];
              if (dom.indexOf(fam) >= 0) return COLOR.aaBlue;
              if (sho.indexOf(fam) >= 0) return COLOR.green;
              return COLOR.aaRed;
            },
            borderRadius: 3,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: function (ctx) { return ctx.parsed.x + " OD-pairs"; } } },
        },
        scales: {
          x: { beginAtZero: true, grid: { color: "#e5e7eb" } },
          y: { grid: { display: false }, ticks: { font: { size: 11 } } },
        },
      },
    });
  }

  // ============== TAB RENDERER ==============

  // Charts only render when their tab becomes visible (Chart.js needs a sized canvas).
  const TAB_CHART_MAP = {
    finetuning: [chartMapeProgression, chartCostComparison, chartLoraPerFamily],
    inference:  [chartInferenceLatency, chartThroughputCost],
    scenarios:  [chartScenarioFrequency],
    dashboard:  [chartCoverageDrift, chartForecastActual, chartPerRouteMape],
    markets:    [chartMarketsRevenueShare, chartMarketsRouteCount, chartRegionMape, chartRegionRevenue],
    "market-routes":   [chartRegionMape, chartRegionRevenue, chartRouteMapeScatter, chartRegionCoverage, populateTopRoutesTable],
    "business-impact": [chartBiRevenueRegion, chartBiMlByRegion, chartBiRolloutTrajectory, chartBiFamilyRevenue],
  };

  const renderedTabs = {};
  window.aaPortfolioRenderTab = function (tabId) {
    if (!TAB_CHART_MAP[tabId] || renderedTabs[tabId]) return;
    // Slight delay so the panel layout stabilises before Chart.js sizes the canvas.
    setTimeout(function () {
      TAB_CHART_MAP[tabId].forEach(function (fn) { fn(); });
      renderedTabs[tabId] = true;
    }, 30);
  };

  // Render the initially-active tab (default overview has no charts).
  document.addEventListener("DOMContentLoaded", function () {
    const active = document.querySelector(".panel.active");
    if (active && active.id) {
      window.aaPortfolioRenderTab(active.id);
    }
  });
})();
