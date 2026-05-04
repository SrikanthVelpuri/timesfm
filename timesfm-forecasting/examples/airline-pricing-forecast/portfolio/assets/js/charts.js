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

  // ============== TAB RENDERER ==============

  // Charts only render when their tab becomes visible (Chart.js needs a sized canvas).
  const TAB_CHART_MAP = {
    finetuning: [chartMapeProgression, chartCostComparison, chartLoraPerFamily],
    inference:  [chartInferenceLatency, chartThroughputCost],
    scenarios:  [chartScenarioFrequency],
    dashboard:  [chartCoverageDrift, chartForecastActual, chartPerRouteMape],
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
