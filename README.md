# TimesFM

TimesFM (Time Series Foundation Model) is a pretrained time-series foundation
model developed by Google Research for time-series forecasting.

*   Paper:
    [A decoder-only foundation model for time-series forecasting](https://arxiv.org/abs/2310.10688),
    ICML 2024.
*   All checkpoints:
    [TimesFM Hugging Face Collection](https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6).
*   [Google Research blog](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/).
*   [TimesFM in BigQuery](https://cloud.google.com/bigquery/docs/timesfm-model):
    an official Google product.

This open version is not an officially supported Google product.

**Latest Model Version:** TimesFM 2.5

**Archived Model Versions:**

-   1.0 and 2.0: relevant code archived in the sub directory `v1`. You can `pip
    install timesfm==1.3.0` to install an older version of this package to load
    them.

## Applied Scientist Portfolio (`docs/`)

A worked example of how TimesFM 2.5 would land in production at airline scale,
written as an Applied Scientist interview portfolio. Open
[`docs/index.html`](docs/index.html) locally for the full reading experience,
or jump straight into a deep-dive below.

<!-- AA-PORTFOLIO-PAGES:START -->

_Auto-generated from `docs/pages/*.html`. Run `python3 scripts/generate_readme_links.py` to refresh, or push to `master` and the [`sync-readme-pages`](.github/workflows/sync-readme-pages.yml) workflow will refresh on your behalf._

**Entry point:** [`docs/index.html`](docs/index.html) — 28 pages.

### Strategy & framing

- **[When to Use TimesFM](docs/pages/when-to-use.html)** — When TimesFM is the right tool: batch vs real-time, cross-sectional vs temporal data. Plus three concrete use cases: setting pricing 14 days ahead, detecting demand shifts in real time, and predicting competitor pricing movements.
- **[Base Price Strategic Layer](docs/pages/base-price-strategy.html)** — The four-layer pricing stack: TimesFM forecast, GBM/Bayesian elasticity, base-price optimizer, dynamic-pricing bot. Why the base price matters more, not less, when bots run the dynamic loop.
- **[Alt-Models Bake-Off](docs/pages/alt-models-bakeoff.html)** — Honest model bake-off for the layers TimesFM does NOT solve: TFT/TiDE/Chronos-Bolt/N-HiTS/DeepAR for forecasting; LightGBM+RE / hierarchical Bayes / Double ML / DeepIV / GPBoost for elasticity; Bayesian opt / contextual bandits for base-price; multi-agent / inverse optimization for competitor reaction.
- **[Market Routes Fine-Tuning](docs/pages/market-routes.html)** — Fine-tuned TimesFM 2.5 across 141 OD-pairs spanning 20 route families: domestic, Mexico, Caribbean, Latin America, Canada, Transatlantic, and Asia long-haul. Per-route MAPE, coverage, and revenue lift.
- **[Business Impact & Revenue](docs/pages/business-impact.html)** — ML metrics translated into business impact: $48M annual revenue lift, 0.62 pp load-factor lift, 9% denied-boarding drop, 27% reduction in analyst overrides. Per-region breakdown across domestic and international markets.

### Model architecture deep-dives

- **[Model Choice: Why TimesFM](docs/pages/model-choice.html)** — Comparison of TimesFM 2.5 against ARIMAX, Prophet, DeepAR, NBEATS, NHITS, Chronos, Lag-Llama, and LLM-with-tools for the AA daily-fare forecasting problem.
- **[TimesFM 2.5 Internals](docs/pages/timesfm-internals.html)** — The decoder-only foundation TS model, opened up. Patching, RoPE, QK-norm attention, continuous quantile head, training corpus, what changed in v2.5 vs v1, and tradeoffs at AA's 6,700-route scale.
- **[Foundation TS Models Survey](docs/pages/foundation-ts-survey.html)** — Architecture-by-architecture deep dive of foundation time-series models: TimesFM 2.5, Chronos-T5, Chronos-Bolt, Moirai, Moirai-MoE, Lag-Llama, TimeGPT, MOMENT, Tiny Time Mixers (TTM). Pretraining corpora, output styles, when to pick each at AA's scale.
- **[Transformer Variants for TS](docs/pages/transformer-variants-ts.html)** — Architecture deep-dive of transformer variants for time-series forecasting: Vanilla Transformer, Informer (ProbSparse), Autoformer (decomposition + autocorrelation), FEDformer (Fourier), PatchTST (patching + channel-independence), Crossformer (two-stage), iTransformer (variate-token), with tradeoffs at scale.

### Technique deep-dives

- **[Fine-Tuning Strategy](docs/pages/finetuning.html)** — LoRA vs QLoRA vs full fine-tune vs adapter modules vs BitFit vs IA-cubed vs frozen-with-linear-probe. PEFT method selection for TimesFM on AA pricing data.
- **[PEFT Methods Deep-Dive](docs/pages/peft-deepdive.html)** — Parameter-efficient fine-tuning methods for foundation models: LoRA, DoRA, QLoRA, IA3, Adapter, Prefix Tuning, P-Tuning v2, LongLoRA, VeRA. Math, target modules, when each wins, and which we ship at AA.
- **[Probabilistic Output Heads](docs/pages/probabilistic-heads.html)** — The probabilistic-forecasting toolbox: continuous quantile heads, parametric distributional heads (Gaussian, Student-T, NegBin, mixtures), discrete-token heads (Chronos), conformal prediction, monotonic CDF parameterizations. Math, calibration, when each wins.
- **[Causal Elasticity Deep-Dive](docs/pages/causal-elasticity-deepdive.html)** — Estimating fare elasticity from observational data: identification strategies (RCT, IV, DiD, RDD, FE), Double ML, DeepIV, causal forests, GPBoost, hierarchical Bayes, and why a forecast coefficient is not an elasticity.
- **[Pricing Optimization Techniques](docs/pages/pricing-optimization.html)** — Optimization techniques for the base-price and dynamic-pricing layers: MIP, Bayesian optimization, contextual bandits (LinUCB, Thompson sampling, NeuralBandit), off-policy evaluation (IPS, DR, SNIPS), reinforcement learning. Math, when each wins, deployment story.
- **[Hierarchical Forecasting & Reconciliation](docs/pages/hierarchical-forecasting.html)** — Forecasting under hierarchies: bottom-up, top-down, middle-out, OLS reconciliation, and MinT (minimum trace). The math, the network -> region -> family -> OD -> cabin hierarchy at AA, and why hierarchical reconciliation buys ~0.6 pp MAPE.

### Production engineering

- **[Inference Optimization](docs/pages/inference.html)** — Layer-by-layer 47x speedup of TimesFM inference: batching, BF16, torch.compile, LoRA hot-swap, KV cache, forecast caching. With code samples and rejected alternatives.
- **[Evaluation Metrics & Calibration](docs/pages/evaluation.html)** — Pinball loss, CRPS, coverage, calibration, reliability diagrams, conformal prediction. How to evaluate probabilistic forecasts for revenue management.
- **[Production Monitoring & Rollback](docs/pages/monitoring.html)** — Drift detection (data, concept, calibration), alerting design, on-call runbook, and one-command rollback path.
- **[Azure Deployment Architecture](docs/pages/azure-architecture.html)** — The Azure deployment topology for the AA pricing forecast service: Azure ML, NV A10 v5 spot VMs, Blob Storage, Cosmos DB, Synapse, AKS, Application Insights. With service mapping and AA-specific compliance considerations.

### Tradeoffs & interview prep

- **[Tradeoffs Deep-Dive](docs/pages/tradeoffs-deepdive.html)** — Every major design decision on the project, why we made it, what we considered, and what we gave up. Cross-cutting view of model choice, fine-tuning, inference, evaluation, deployment, and monitoring tradeoffs.
- **[Interview Q&A by Section](docs/pages/interview-qa.html)** — Comprehensive interview prep: 100+ questions across foundations, model choice, fine-tuning, inference, evaluation, deployment, monitoring, market generalisation, and business impact. Each Q with a concrete A grounded in the project.
- **[Interview Cheat Sheet](docs/pages/cheatsheet.html)** — One-page quick-reference of the AA pricing forecast project: numbers, math, code, decision rules, killer Q&A, and a glossary. Built for the night before an interview.

### Scenarios

- **[Scenario: Thanksgiving Rush](docs/pages/scenario-holiday.html)** — How TimesFM handles the Thanksgiving holiday peak: covariate inputs, forecast shape, bid-price response, and the trade-offs of aggressive vs conservative q90.
- **[Scenario: Fuel-Price Spike](docs/pages/scenario-fuel.html)** — A 9% Brent crude opening forces the entire network to re-price. How TimesFM's fuel-pass-through coefficient, off-cycle re-scoring, and SLO-driven freshness handle the shock.
- **[Scenario: Competitor Undercut](docs/pages/scenario-competitor.html)** — Delta drops fares on ATL-LAX overnight. How TimesFM's competitor-fare covariate, family-specific elasticity, and a 14-minute off-cycle latency drive AA's response on adjacent routes.
- **[Scenario: Weather Irrops](docs/pages/scenario-weather.html)** — A winter storm warning at ORD flags weather-disruption days. How TimesFM avoids feedback loops by training on scheduled (not realized) irrops, and why the -$22 coefficient is intentional.
- **[Scenario: New Route Cold-Start](docs/pages/scenario-coldstart.html)** — JFK-PVG launches with zero history. Synthetic-prior bootstrapping, foundation-model zero-shot, hierarchical pooling, and how to forecast a route that has never flown.
- **[Scenario: Cabin Upsell](docs/pages/scenario-cabin.html)** — Forecasting premium-economy fares on DFW-LAX. Why the wider PI is correct, how static-covariate interactions work, and what it takes to support a cabin-mix simulation.

<!-- AA-PORTFOLIO-PAGES:END -->

## Update - Mar. 19, 2026

Huge shoutout to [@borealBytes](https://github.com/borealBytes) for adding the support for [AGENTS](https://github.com/google-research/timesfm/blob/master/AGENTS.md)! TimesFM [SKILL.md](https://github.com/google-research/timesfm/tree/master/timesfm-forecasting) is out.

## Update - Oct. 29, 2025

Added back the covariate support through XReg for TimesFM 2.5.


## Update - Sept. 15, 2025

TimesFM 2.5 is out!

Comparing to TimesFM 2.0, this new 2.5 model:

-   uses 200M parameters, down from 500M.
-   supports up to 16k context length, up from 2048.
-   supports continuous quantile forecast up to 1k horizon via an optional 30M
    quantile head.
-   gets rid of the `frequency` indicator.
-   has a couple of new forecasting flags.

Along with the model upgrade we have also upgraded the inference API. This repo
will be under construction over the next few weeks to

1.  add support for an upcoming Flax version of the model (faster inference).
2.  add back covariate support.
3.  populate more docstrings, docs and notebook.

### Install

1.  Clone the repository:
    ```shell
    git clone https://github.com/google-research/timesfm.git
    cd timesfm
    ```

2.  Create a virtual environment and install dependencies using `uv`:
    ```shell
    # Create a virtual environment
    uv venv
    
    # Activate the environment
    source .venv/bin/activate
    
    # Install the package in editable mode with torch
    uv pip install -e .[torch]
    # Or with flax
    uv pip install -e .[flax]
    # Or XReg is needed
    uv pip install -e .[xreg]
    ```

3. [Optional] Install your preferred `torch` / `jax` backend based on your OS and accelerators
(CPU, GPU, TPU or Apple Silicon).:

-   [Install PyTorch](https://pytorch.org/get-started/locally/).
-   [Install Jax](https://docs.jax.dev/en/latest/installation.html#installation)
    for Flax.

### Code Example

```python
import torch
import numpy as np
import timesfm

torch.set_float32_matmul_precision("high")

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")

model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[
        np.linspace(0, 1, 100),
        np.sin(np.linspace(0, 20, 67)),
    ],  # Two dummy inputs
)
point_forecast.shape  # (2, 12)
quantile_forecast.shape  # (2, 12, 10): mean, then 10th to 90th quantiles.
```
