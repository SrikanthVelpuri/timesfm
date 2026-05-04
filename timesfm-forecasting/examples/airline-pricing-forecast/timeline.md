# Project Timeline &mdash; AA Pricing Forecast with TimesFM

A 16-week project broken into 8 two-week sprints. Each sprint maps to one
of the EPICs in [user_stories.md](user_stories.md). The schedule below is
the one I held myself to as the Applied Scientist (acting as ML Engineer)
running the project end-to-end.

---

## At a glance

```
Sprint:   1     2     3     4     5     6     7     8
Week:     1-2   3-4   5-6   7-8   9-10  11-12 13-14 15-16

         | Discovery   |     |     |     |     |     |     |
         |     | Data + Eval |     |     |     |     |     |
         |     |     | Zero-shot  |     |     |     |     |
         |     |     |     | XReg head |     |     |     |
         |     |     |     |     | LoRA      |     |     |
         |     |     |     |     |     | Inference |     |
         |     |     |     |     |     |     | Canary    |
         |     |     |     |     |     |     |     |Rollout
```

| Sprint | Weeks | Theme | Key deliverable | Risk |
| ------ | ----- | ----- | --------------- | ---- |
| 1 | 1&ndash;2 | Discovery | Baseline memo + go/no-go | Low |
| 2 | 3&ndash;4 | Data + Eval Harness | Airflow DAG + eval module | Medium |
| 3 | 5&ndash;6 | Zero-shot Baseline | Network MAPE on val cohort | Low |
| 4 | 7&ndash;8 | XReg Head Fine-tune | `aa-timesfm-2.5-xreg-v1` | Medium |
| 5 | 9&ndash;10 | LoRA Adapters | 17 family adapters | Medium |
| 6 | 11&ndash;12 | Inference Optimization | 41 s nightly run | High |
| 7 | 13&ndash;14 | Canary | 5% rollout, sign-off | High |
| 8 | 15&ndash;16 | Full Rollout | 100% on production | High |

---

## Sprint 1 &mdash; Discovery (weeks 1&ndash;2)

**Goal.** Decide if the project is worth running at all.

**Activities**

- W1: Pull 6 months of fares for 50 representative OD-pairs. Profile
  distributions, identify gotchas (cabin mix, currency, fare basis).
- W1: Inventory the 47 features the production ARIMAX uses; classify into
  TimesFM&apos;s {static, dynamic-numerical, dynamic-categorical} taxonomy.
- W2: Reproduce the production ARIMAX baseline on the discovery sample.
- W2: Run TimesFM 2.5 zero-shot on the same sample.
- W2: Write 3-page memo with go/no-go recommendation.

**Exit gate**

- Director of RM signs the memo.
- Decision recorded: go (zero-shot was within 1.5 pp MAPE of ARIMAX
  *before* any fine-tuning &mdash; high confidence the fine-tuned version
  beats it convincingly).

**My role**

- Half my time: literally writing SQL against the AAdvantage warehouse.
  Not glamorous; it&apos;s how I learned the data&apos;s shape.
- Other half: the modeling notebook + the memo writing.

---

## Sprint 2 &mdash; Data + Evaluation Harness (weeks 3&ndash;4)

**Goal.** Stand up the infrastructure that every later sprint depends on.

**Activities**

- W3: Write the Airflow DAG that lands the daily feature panel in S3.
  Schema-version it. DataHub lineage.
- W3: Write the eval module (`aa_pricing_eval/`). Pinball loss, MAPE,
  sMAPE, calibration, residual decomposition.
- W4: Backfill 24 months of features. Verify idempotency by re-running.
- W4: Wire eval module into CI; any PR that touches model code runs it
  on a frozen mini-cohort.
- W4: First version of the calibration drift dashboard.

**Exit gate**

- Airflow DAG runs successfully for 7 consecutive days.
- Eval module produces identical numbers in CI as on my laptop.
- Dashboard shows real coverage on the 50-pair discovery cohort.

**My role &mdash; the ML Engineer pickup**

- This is the sprint where I stopped being a research scientist. I&apos;d
  never written an Airflow DAG before. I broke prod (well, dev) twice on
  day-2 retries before figuring out poll-with-backoff for the upstream
  Sabre file. The kind of mistake you make exactly once.

**Risks**

- *Risk: upstream Sabre extract is unreliable.* Mitigation: 6-hour SLA
  buffer, fallback to last-good extract.
- *Risk: schema breaks in the AAdvantage warehouse.* Mitigation: contract
  tests on schema; alert on breaking changes.

---

## Sprint 3 &mdash; Zero-shot Baseline (weeks 5&ndash;6)

**Goal.** Establish the number every later sprint has to beat.

**Activities**

- W5: Run TimesFM 2.5 zero-shot across the full 24-month val cohort
  (~3,500 stable OD-pairs).
- W5: Per-route-family breakdown. Identify families where zero-shot
  underperforms ARIMAX.
- W6: Calibration analysis. Are the quantiles honest?
- W6: 2-page summary doc, baseline number locked.

**Exit gate**

- Network MAPE: 10.2% (target was &le; 11%).
- 80% PI coverage: 79.0% (target was 78&ndash;82%).
- Per-family results documented.
- Baseline number frozen for the rest of the project &mdash; "anything
  later has to beat 10.2%".

**My role**

- All modeling. This was the most "Applied Scientist-feeling" sprint of
  the project &mdash; just running the model and analyzing residuals.

---

## Sprint 4 &mdash; XReg Head Fine-tuning (weeks 7&ndash;8)

**Goal.** Train the AA-tuned regression head &mdash; the cheapest, biggest
accuracy lever in the whole project.

**Activities**

- W7: Set up the training script (frozen trunk, only the head trains).
  Pinball + MSE composite loss.
- W7: Hyperparameter sweep over LR, weight decay, rank of the cross-route
  interaction term. ~20 runs total.
- W8: Best-config training run. Sanity-check the learned coefficients
  against Operations Research team&apos;s independent calculations.
- W8: Sign and publish `aa-timesfm-2.5-xreg-v20251201.safetensors`.

**Exit gate**

- Val MAPE: 6.0% (down from 10.2%; target was &le; 6.5%).
- Coefficients within ~10% of OR team&apos;s priors. (They were within
  8.4%.)
- Artifact signed, published, lineage in DataHub.

**My role**

- Modeling + a Systems Engineer detour: I had to learn cosign so the
  artifact had a verifiable signature. Ended up writing a wrapper script
  in `scripts/sign_artifact.py` because `cosign` CLI ergonomics are not
  great. That script became standard for the team.

---

## Sprint 5 &mdash; LoRA Adapters (weeks 9&ndash;10)

**Goal.** Ship per-route-family adapters for the top 200 OD-pairs.

**Activities**

- W9: Group top-200 OD-pairs into 20 route families. Document the cluster
  definitions for downstream consumers.
- W9: LoRA training loop. Rank sweep {4, 8, 16}; rank 8 wins.
- W10: Train all 20 families. ~6 minutes each on A10G.
- W10: Per-family eval. 17 ship; 3 fall back to head-only.
- W10: Build the `aa_lora_registry/` Python module that maps OD-pair
  &rarr; adapter file.

**Exit gate**

- Network MAPE: 5.1% (target was &le; 5.5%).
- Top-200 MAPE: 4.4%.
- 17 of 20 families pass; 3 fail and fall back gracefully.
- Total Stage B compute cost: $6.50 (target was &lt; $25).

**My role**

- Modeling + adapter loading protocol design (the Systems Engineer-y
  part). The naive load-from-disk version was 800 ms; I had to design
  the pinned-memory hot-swap path in this sprint to know if rank-8 was
  actually viable in production.

**Risks**

- *Risk: LoRA at rank-8 is too constrained to capture family-specific
  patterns.* Mitigated by the rank sweep showing rank-8 was within 0.1
  pp of rank-16.

---

## Sprint 6 &mdash; Inference Optimization (weeks 11&ndash;12)

**Goal.** Hit the 02:30 CST nightly SLO with margin.

**Activities**

- W11: Profile baseline. Build the optimization-stack table.
- W11: Apply batching, BF16, `torch.compile` in sequence; measure each.
- W11: Sweep batch size; pick 64.
- W12: LoRA hot-swap with pinned memory.
- W12: Forecast caching with covariate-quantization-aware hashing.
- W12: Final profiler pass to confirm no FP32 stragglers, KV cache
  reuse working, no recompilation in steady state.

**Exit gate**

- Wall time: 41 s (target was &lt; 60 s).
- p99 per-OD latency: 14 ms (target was &lt; 20 ms).
- Cost per run: $0.011 (target was &lt; $0.10).
- See [inference_optimization.md](inference_optimization.md) for the full
  breakdown.

**My role**

- 100% ML Engineer in this sprint. PyTorch Profiler, CUDA graphs,
  pinned memory, JIT compilation. I had not used any of these
  professionally before. By the end of the sprint they were standard
  tools.

**Risks**

- *Risk: `torch.compile` recompilation in steady state.* Mitigated by
  pad-to-64 strategy.
- *Risk: BF16 calibration drift.* Verified to be zero-impact via
  side-by-side hold-out.

---

## Sprint 7 &mdash; Canary (weeks 13&ndash;14)

**Goal.** Prove the system in production on a tiny surface before exposing
real revenue.

**Activities**

- W13: Wire the model into the feature store on a separate namespace
  (shadow mode).
- W13: Daily diff report comparing TimesFM forecasts to ARIMAX for 5
  low-revenue OD-pairs.
- W13: Synthetic on-call drill (got paged on a Saturday, resolved
  simulated regression in 6 m 12 s).
- W14: Director sign-off on the canary results.
- W14: Expand to 5% of OD-pairs (~335 routes), still in shadow mode.

**Exit gate**

- 7 days clean on canary, no model failures.
- Coverage maintained at 79&plusmn;3% on canary set.
- Director signs off.
- 5% expansion runs successfully for 7 more days.

**My role**

- Stakeholder communication and runbook authoring became the dominant
  workload. Modeling was already done; this was about packaging the
  result for people who don&apos;t read PyTorch.

---

## Sprint 8 &mdash; Full Rollout (weeks 15&ndash;16)

**Goal.** Take the system to 100% of OD-pairs with the existing
optimizer reading TimesFM forecasts as the source of truth.

**Activities**

- W15: 5% &rarr; 25% &rarr; 50% staged rollout, 2 days at each stage.
- W15: PagerDuty alerts wired for coverage drift, runtime breach,
  cost overrun.
- W15: Rollback runbook published, peer-reviewed by the ML Eng team.
- W16: 50% &rarr; 75% &rarr; 100%.
- W16: Wrap-up doc; hand-off-light to ML Engineering for shared on-call
  ownership.

**Exit gate**

- 100% rollout complete, SLOs held.
- Network MAPE in production: 5.1% (matches val cohort number).
- Estimated revenue lift validated by Pricing team A/B analysis.
- On-call rotation in place: me primary, ML Eng secondary.

**My role**

- Mostly writing: runbooks, hand-off doc, post-mortem template, the
  weekly status update for VP. Modeling was wrapped; this was the
  packaging-for-other-humans work.

---

## What I would do differently

If I were running this project again:

1. **Sprint 2 was too short.** The eval harness was the highest-leverage
   piece of code in the whole project, and I rushed it. Next time I&apos;d
   give it 3 weeks and skip a week from Sprint 8 (which I overscoped
   anyway).

2. **Stakeholder cadence too late.** I started weekly directors updates
   in Sprint 5. Should have started in Sprint 1 &mdash; even just a 5-line
   email. Without it, the Sprint-4 sign-off conversation took longer than
   it needed to because the director had to catch up on everything.

3. **Synthetic on-call drill earlier.** I did it in Sprint 7. Should have
   done a paper drill in Sprint 4. The first time I thought hard about
   what an alert would look like was way too late.

4. **More aggressive scope cuts.** I shipped 17 LoRA adapters; the bottom
   5 contributed almost nothing to the network MAPE. Could have shipped
   12 and saved a week.

The point of writing this down is that I learned more from things 2 and
3 than from anything else in the project. The technical work was
straightforward by comparison.

---

## What this timeline does NOT include

- **No continued pretraining.** AA had no budget for it; the timeline
  reflects that. See [finetuning_aa_data.md](finetuning_aa_data.md) for
  the rationale.
- **No multi-model ensemble.** Considered briefly in Sprint 3; rejected
  because it doubles inference cost for &lt; 0.3 pp MAPE gain.
- **No real-time online serving.** Out of scope; the project is
  forecast-as-a-feature for a batch optimizer.

If the program ever wants those, they would each be a separate
multi-sprint track.

---

## Portfolio link

The interactive version of this timeline (with charts of MAPE
progression, latency reduction, cost over time) is in
[portfolio/index.html](portfolio/index.html), tab "Timeline".
