# User Stories &mdash; Applied Scientist Acting as End-to-End ML Engineer

> **Who I am.** I am an Applied Scientist on the AA Revenue Management AI
> team. My job description says "design models, write papers, hand the code
> off to ML Engineering." For this project, the ML Engineering org was
> headcount-constrained and the team lead asked me to **own the entire
> lifecycle** &mdash; data extraction, modeling, evaluation, inference
> optimization, canary deployment, monitoring, and on-call.
>
> I had to step out of notebook mode and act like an ML Engineer. These user
> stories are how I structured that work. Each one is the ticket I would
> have written for myself in Jira, with the acceptance criteria I held
> myself to and the artifacts I shipped.

This doc is the narrative companion to the [forecast_aa_pricing.py](forecast_aa_pricing.py)
example. The 16-week schedule is in [timeline.md](timeline.md). The
production inference path is in [inference_optimization.md](inference_optimization.md).

---

## How my role expanded

| Phase | "Applied Scientist" job | "ML Engineer" job I picked up |
| ----- | ----------------------- | ----------------------------- |
| Data | Pull a clean sample, profile distributions | Build the production extract job, schema-version the feature store |
| Model | Pick the right architecture, write the loss | Containerize, version artifacts, sign safetensors |
| Eval | MAPE / coverage in a notebook | Reusable eval module, CI gate, calibration drift dashboard |
| Inference | Verify accuracy on a held-out slice | BF16 + `torch.compile` + LoRA hot-swap; hit the latency SLO |
| Deploy | Email a notebook | Canary &rarr; 5% &rarr; 25% &rarr; full, with auto-rollback |
| Monitor | Look at residuals on Friday | Coverage drift alerts paging me at 2 a.m. |

The point is **none of this required new skills I didn&apos;t already
have** &mdash; it required me to *apply* them outside the comfort zone of a
Jupyter notebook. Every story below is one chunk of that translation.

---

## EPIC 1 &mdash; Discovery and scoping (Sprint 1: weeks 1&ndash;2)

### Story 1.1 &mdash; Make the case for a foundation model

> **As an Applied Scientist**
> **I want to** quantify how much accuracy we leave on the table with the
> per-route ARIMAX baseline,
> **so that** I can justify (or kill) a foundation-model approach before we
> spend a quarter on it.

**Acceptance criteria**

- Pull 6 months of daily fares for a stratified sample of 50 OD-pairs (10 hub-corp, 20 leisure, 10 transcon, 10 international).
- Reproduce the current production ARIMAX forecast on the same sample.
- Run TimesFM 2.5 zero-shot on the same sample.
- Compare MAPE, sMAPE, 80% PI coverage, and Q4-only MAPE.
- Write a 3-page memo with the result and a go/no-go recommendation.

**Artifact shipped**

- `notebooks/01_baseline_comparison.ipynb` &mdash; reproducible by anyone
  on the team with a Sabre extract.
- `memos/timesfm_go_decision.md` &mdash; the go-decision the director
  signed off on.

**ML-Engineer skill I had to pick up**

- Building a *reproducible* notebook with pinned env, data hash, and
  seed control &mdash; not the throwaway research notebook I was used to.
  Future me would re-run this every quarter; it had to work without
  archeology.

---

### Story 1.2 &mdash; Map the AA covariate catalog onto the TimesFM XReg API

> **As an Applied Scientist**
> **I want to** confirm that every covariate the RM team uses today fits
> into TimesFM&apos;s {static categorical, dynamic numerical, dynamic
> categorical} taxonomy,
> **so that** I don&apos;t discover a missing feature class in week 12.

**Acceptance criteria**

- Inventory every feature in the production ARIMAX (47 features total).
- Classify each into the TimesFM taxonomy.
- Flag features that don&apos;t fit cleanly &mdash; document workarounds.
- Reach agreement with the Pricing team on which features are
  must-have for v1.

**Outcome**

- 8 dynamic + 3 static covariates make v1 (the same set in
  [forecast_aa_pricing.py](forecast_aa_pricing.py)).
- 12 features deferred to v2 (mostly inventory-shape features that
  require a separate forecasting head).
- 27 features dropped &mdash; correlated with v1 features or low SHAP
  importance in the ARIMAX baseline.

**ML-Engineer skill I had to pick up**

- Negotiating scope. Saying "no" to a feature is harder than adding it.
  I had to defend the v1 cut against three stakeholders who wanted their
  pet feature in scope.

---

## EPIC 2 &mdash; Data and evaluation infrastructure (Sprint 2: weeks 3&ndash;4)

### Story 2.1 &mdash; Stand up a production-grade extract job

> **As an Applied Scientist**
> **I want** a daily 03:00 CST job that lands a feature-engineered panel
> in S3,
> **so that** model training and nightly inference both read from one
> source of truth.

**Acceptance criteria**

- Airflow DAG with retries, alerting, and lineage in DataHub.
- Output schema: parquet partitioned by departure date, &lt; 1 GB/day.
- Schema is versioned; breaking changes require a major version bump.
- Backfill: 24 months of history, idempotent, completes in &lt; 6 hours.
- Unit tests for every transform; integration test on a frozen day.

**Artifact shipped**

- `airflow/dags/aa_pricing_features_dag.py`
- `aa_pricing_features/` Python package (transforms, schemas, tests)
- DataHub dataset `aa_pricing.daily_features.v1`

**ML-Engineer skill I had to pick up**

- Writing Airflow DAGs that actually survive on-call. My first version
  blew up at 03:14 a.m. on day one because Sabre publishes its file at
  03:09 and I had a brittle existence-check. I rewrote with poll-with-
  backoff. That kind of failure mode was not on my Applied Scientist
  bingo card.

---

### Story 2.2 &mdash; Build the evaluation harness

> **As an Applied Scientist**
> **I want** a reusable evaluation module that any new model checkpoint
> can be plugged into,
> **so that** experiments are comparable and the eval is the same in CI as
> on my laptop.

**Acceptance criteria**

- Python package `aa_pricing_eval/` with a single entry point:
  `evaluate(model, dataset_id, output_dir)`.
- Outputs: `metrics.json`, calibration plots, residual decomposition,
  per-route-family breakdown.
- Honors temporal splits; refuses to run if any horizon point has a
  context leak.
- Runs in CI on every PR that touches the model code.

**Artifact shipped**

- `aa_pricing_eval/` package (eval, calibration, plotting, leakage
  guards).
- `.github/workflows/model_eval.yml` &mdash; CI gate.
- `dashboards/calibration_drift.html` &mdash; rolling 30-day calibration
  for production.

**ML-Engineer skill I had to pick up**

- Treating the eval as code, not as a notebook. Every metric I report
  has a unit test for the math. That sounds excessive until the day a
  reviewer asks "is your MAPE weighted by route revenue or unweighted?"
  and you can answer in 30 seconds with a test that proves which.

---

## EPIC 3 &mdash; Modeling: zero-shot baseline (Sprint 3: weeks 5&ndash;6)

### Story 3.1 &mdash; Establish the zero-shot baseline

> **As an Applied Scientist**
> **I want to** measure TimesFM 2.5 zero-shot on the full AA panel (not
> just the discovery sample),
> **so that** every later improvement is grounded against a real number.

**Acceptance criteria**

- Run zero-shot on the full 24-month val cohort across all stable
  OD-pairs (~3,500 series).
- Report MAPE, sMAPE, 80% PI coverage, Q4 MAPE, holiday-day MAPE,
  per-route-family breakdowns.
- Identify the top three route families where zero-shot is *worse* than
  ARIMAX.
- Write up findings in a 2-page summary.

**Outcome**

- Zero-shot MAPE: 10.2% network, 14.1% on holiday days.
- Per-route-family results: zero-shot beat ARIMAX on 16 of 20 families;
  worse on 4 (transatlantic premium, deep-South leisure, two niche
  international).
- Conclusion: foundation model is a strict improvement on the long tail
  and the cold-start; underperforms on routes with strong AA-specific
  patterns &rarr; case for fine-tuning is real.

**ML-Engineer skill I had to pick up**

- Running batch jobs at scale on a budget. First attempt cost $90 because
  I forgot to set `per_core_batch_size`. Set it to 64 on A10G, the same
  job cost $0.40. Now I check batch size before any wall-clock-sensitive
  job; muscle memory I didn&apos;t have before.

---

## EPIC 4 &mdash; Modeling: XReg head fine-tuning (Sprint 4: weeks 7&ndash;8)

### Story 4.1 &mdash; Fine-tune the XReg head on AA covariates

> **As an Applied Scientist**
> **I want to** train a regression head that maps AA covariates to fare
> residuals,
> **so that** the foundation model can incorporate AA-specific elasticities
> we know but it doesn&apos;t.

**Acceptance criteria**

- Trunk frozen; only the regression head trains.
- Pinball loss across all 10 quantiles + median MSE, equally weighted.
- Trained on 250K windows from the train cohort; eval on val cohort.
- Beats zero-shot by &ge; 3 percentage points MAPE on val cohort.
- Coefficients pass the domain-prior sanity check (within ~10% of the
  OR team&apos;s independent calculations).

**Result**

- MAPE 6.0% on val (down from 10.2% zero-shot).
- Holiday-day MAPE 8.7% (down from 14.1%).
- Coverage 79.3% (target 78&ndash;82%).
- Coefficients sanity-checked by Operations Research team &mdash; all
  within tolerance.

**Artifact shipped**

- `aa-timesfm-2.5-xreg-v20251201.safetensors` (~1.2 MB).
- Training script, config, signed lineage record in the model registry.

**ML-Engineer skill I had to pick up**

- Versioning model artifacts with signatures. I didn&apos;t know what
  cosign was at the start of the project. Now every artifact I ship has
  a signature tied to the training data hash and my engineer key.

---

## EPIC 5 &mdash; Modeling: LoRA adapters (Sprint 5: weeks 9&ndash;10)

### Story 5.1 &mdash; Train per-route-family LoRA adapters for top-200 OD-pairs

> **As an Applied Scientist**
> **I want to** capture AA-specific cyclic patterns on high-revenue routes
> through cheap LoRA adapters,
> **so that** I can squeeze out the last 0.5&ndash;1 percentage point of
> MAPE without retraining the trunk.

**Acceptance criteria**

- Group top-200 OD-pairs into 20 route families.
- Train rank-8 LoRA adapter per family on attention Q/K/V/O projections.
- Each family ships only if its LoRA-adapted MAPE beats the head-only
  MAPE on the same family by &ge; 0.5 pp.
- Failed families fall back to head-only forecast (no adapter).
- Total compute budget: &le; 8 GPU-hours on A10G.

**Result**

- 17 of 20 families ship with adapters; 3 fall back to head-only.
- Network MAPE 5.1% (down from 6.0% head-only).
- Top-200 MAPE 4.4% (down from 5.6% head-only on those routes).
- Total Stage B compute: 6.3 GPU-hours, cost $6.50.

**Artifact shipped**

- 17 LoRA `.safetensors` files (~4 MB each), keyed by route family.
- `aa_lora_registry/` Python module that maps OD-pair &rarr; adapter.

**ML-Engineer skill I had to pick up**

- Designing the *adapter loading protocol*. The naive version (load
  adapter from S3 per OD-pair) was 800 ms per swap &mdash; unusable.
  Final version: pre-load all 17 into pinned CPU memory at startup and
  do `state_dict.update()` on the resident GPU model. Got it to 8 ms
  amortized per swap. That is a Systems-Engineer-y problem and I had to
  learn it on the fly.

---

## EPIC 6 &mdash; Inference optimization (Sprint 6: weeks 11&ndash;12)

### Story 6.1 &mdash; Hit the nightly batch SLO

> **As an Applied Scientist (acting as ML Engineer)**
> **I want** the nightly batch over 6,700 OD-pairs to complete in &lt; 60
> seconds wall time on a single A10G,
> **so that** the C++ optimizer reads fresh forecasts before 02:30 CST and
> we leave headroom for re-runs.

**Acceptance criteria**

- BF16 mixed precision enabled.
- `torch.compile` on inference graph.
- Optimal batch size determined empirically.
- LoRA hot-swap path benchmarked at &lt; 10 ms.
- Wall time &lt; 60 s for 6,700 series.
- p99 per-OD-pair latency &lt; 20 ms.

**Result &mdash; per-OD-pair median latency, A10G**

| Configuration | Latency | Throughput |
| ------------- | ------- | ---------- |
| FP32, batch=1, eager | 280 ms | 3.6 / s |
| + batch=64 | 18 ms | 56 / s |
| + BF16 | 9 ms | 110 / s |
| + `torch.compile` | 6 ms | 165 / s |
| + LoRA hot-swap (final) | 6 ms median, 14 ms p99 | 165 / s |

Wall time for 6,700 OD-pairs: **41 seconds** &mdash; comfortably inside
the 60s SLO.

The full breakdown of every optimization is in
[inference_optimization.md](inference_optimization.md).

**ML-Engineer skill I had to pick up**

- Using PyTorch Profiler. I had never opened it before this project.
  Now I run it on every checkpoint to confirm the kernel mix is what I
  expect. Caught a stray FP32 kernel in my BF16 graph the first
  week &mdash; would have eaten 30% of my throughput budget if I
  hadn&apos;t checked.

---

## EPIC 7 &mdash; Deployment and monitoring (Sprint 7&ndash;8: weeks 13&ndash;16)

### Story 7.1 &mdash; Canary deploy to 5 OD-pairs

> **As an Applied Scientist (acting as ML Engineer)**
> **I want** to canary the foundation-model forecast on 5 low-revenue
> OD-pairs in shadow mode for 7 days,
> **so that** I can compare its outputs to the production ARIMAX without
> any customer-facing risk.

**Acceptance criteria**

- Shadow mode: model writes to a separate feature-store namespace; the
  C++ optimizer continues reading ARIMAX forecasts.
- Daily diff report comparing the two forecasts.
- Coverage of 80% PI on these 5 routes maintained at 78&ndash;82%.
- No model failures, OOMs, or deployment rollbacks.
- Sign-off from Director of RM before expanding scope.

**Result**

- 7 days clean. Daily diff report shows the two forecasts within the
  expected uncertainty band on 6 of 7 days; the seventh day was a
  weather-irrops day where TimesFM correctly flagged a -$22 dip that
  ARIMAX missed.
- Director signed off; expanded to 5% of OD-pairs in week 14.

**ML-Engineer skill I had to pick up**

- Writing a diff report that a director can read. My first version was
  a 12-tab spreadsheet; nobody opened it. Final version: a 1-page PDF
  with three charts and one go/no-go recommendation. That is a
  communication skill, not a modeling skill, but it was on the critical
  path.

---

### Story 7.2 &mdash; Wire production monitoring and rollback

> **As an Applied Scientist (acting as ML Engineer)**
> **I want** automatic alerts on coverage drift and a one-command
> rollback path,
> **so that** I can be paged at 2 a.m. and have the service back to a
> known-good state in &lt; 5 minutes.

**Acceptance criteria**

- PagerDuty alert if rolling 7-day 80% PI coverage drops below 75% on
  any top-50 route family.
- PagerDuty alert if nightly run misses the 02:30 CST deadline.
- One-command rollback: `aa_pricing_cli rollback --stage [head|lora] --to <run_id>`.
- Documented runbook for the on-call engineer.
- Synthetic on-call drill: I get paged on a Saturday and resolve a
  simulated regression in &lt; 10 minutes.

**Outcome**

- Synthetic drill ran in week 15, resolved in 6 minutes 12 seconds.
- First real page came in week 19 (post-launch): a competitor-fare feed
  partial outage caused a coverage dip on 3 leisure routes. Rolled the
  head adapter back to the previous week&apos;s checkpoint, paged the
  data team about the upstream feed, root-cause was upstream within 22
  minutes. Rollback worked exactly as the runbook said.

**ML-Engineer skill I had to pick up**

- Writing runbooks. As an Applied Scientist, my "documentation" was a
  paper. As an on-call engineer, my "documentation" is the runbook
  someone half-asleep at 2 a.m. has to read while a director is
  refreshing the dashboard. Different audience, different doc, same
  underlying knowledge.

---

### Story 7.3 &mdash; Full network rollout

> **As an Applied Scientist (acting as ML Engineer)**
> **I want to** ramp the new forecast service from 25% to 100% of
> OD-pairs over two weeks,
> **so that** any unforeseen issues affect a small surface first.

**Acceptance criteria**

- 25% &rarr; 50% &rarr; 75% &rarr; 100% over 14 days.
- Coverage SLO held at every step.
- Runtime SLO held at every step.
- Cost per nightly run &lt; $25 (turned out to be $22 at 100%).
- Communication plan: weekly update to Director, biweekly to VP.

**Result**

- Hit 100% on schedule.
- Network MAPE: 5.1% (down from 7.4% ARIMAX baseline &mdash; a 31%
  relative improvement).
- Estimated annual revenue lift: ~$48M (validated post-hoc by a
  Pricing team A/B analysis on the staggered rollout).
- Service has been in production for 3 quarters as of writing,
  ~zero-touch operations.

---

## What this taught me about being an Applied Scientist who can ship

The most useful thing I learned is that the **boundary between Applied
Scientist and ML Engineer is artificial**. The real boundary is between
"work that ends in a notebook" and "work that ends in a system another
human can rely on." I crossed that boundary by:

1. **Treating my eval, my data pipeline, and my training scripts as
   code** &mdash; tested, reviewed, versioned. Not artifacts of a
   research process; deliverables in their own right.

2. **Optimizing for the failure modes I would have to debug at 2 a.m.**
   &mdash; idempotency, backfill, retries, signed artifacts, runbook,
   rollback. Stuff that has zero presence in a paper but every presence
   on-call.

3. **Building stakeholder trust through reproducibility.** Every number
   I reported came with a notebook and a frozen dataset hash. When the
   Director asked "what changes if we exclude COVID from training?" I
   could answer in an hour, not a week, because the harness was already
   built.

4. **Negotiating scope ruthlessly.** v1 had 8 covariates, not 47. v1
   had 200 OD-pairs with LoRA, not all 6,700. v1 was nightly batch, not
   real-time. Each cut was a fight, but each was the right cut for the
   timeline.

5. **Owning the boring parts of MLOps.** The CI gate, the rollback
   command, the runbook, the alerts &mdash; those are where the project
   would have died if I&apos;d waited for ML Engineering to pick them up.
   I picked them up because nobody else was going to.

This is the kind of project I&apos;d look back on and say &mdash; this is
where I stopped being a researcher who happens to know ML and started
being an applied scientist who can ship production ML end-to-end.

The portfolio site at [portfolio/index.html](portfolio/index.html) is the
public-facing version of this story, with interactive charts for the
results and a tabbed view of every artifact.
