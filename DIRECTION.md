# Can a linear probe tell, at the moment an error is written, whether the model will catch it?

**One-line claim being tested.** When a wrong value enters a model's chain of thought, the residual
stream at that token already encodes whether the model will *follow* the error to a wrong answer or
*catch* it, and this signal carries information beyond the problem's difficulty, the kind and size of
the error, and the model's own surprise at the wrong number.

**Why it matters for safety.** CoT monitoring assumes the written chain drives the answer. Content
the model did not author (tool outputs, retrieved text, sub-agent traces, edited reasoning) enters
the chain constantly. A monitor needs to know, per step, whether a bad value will be absorbed
silently. That is a per-instance decision a linear probe can make in microseconds, and whose
correctness we can score exactly.

---

## 1. Protocol lineage (what we borrow, and from whom)

The overthinking project co-opted the *forced-answer / probe-gated early exit* procedure of
Reasoning Theater. This project co-opts two established procedures in the same way:

| Component | Borrowed from | What they did | What we change |
|---|---|---|---|
| **Measurement: perturb one step, continue** | Lanham et al. 2023, *Measuring Faithfulness in CoT* ("adding mistakes"); Jia & Mu 2026, *From Decorative to Load-Bearing* (continuation-based causal testing) | Insert a mistake into a CoT step, truncate, let the model continue, see whether the answer follows | Mistakes are **deterministic** edits of verified equations; the FOLLOW target is **recomputed symbolically** from the model's own equations, so no LLM judge anywhere |
| **Error types and sizes** | Knowing-Saying Gap 2026 (typed synthetic errors); Jia & Mu 2026 (magnitude sweep); Phan et al. 2026 (injected arithmetic slips) | Fixed error types (off-by-one, ×10, operator) | Six operators, three of them **random** so error size and plausibility vary continuously and "which operator" stops being a shortcut |
| **Intervention: "Wait"** | Tsui 2025, *Self-Correction Bench* | Appending "Wait" after an error activates correction (−89% blind spot) | Used as the **action** a monitor can take; generated for every corrupted *and* clean step so its benefit and its cost are both measured |
| **Probe-gated action** | Boppana et al. 2026, *Reasoning Theater* | A probe decides when to stop a trace (early exit) | The probe decides **where** to spend a limited "Wait" budget; compared against surprisal-, covariate-, random- and oracle-ranked policies |
| **Causal test** | Jia & Mu 2026 (steering fails); Phan et al. 2026 (critique vector works) | Additive steering along a probe/difference direction | Same, with random-direction controls, on held-out catchable errors |

## 2. Datasets

| Dataset | n | Why | Used by |
|---|---|---|---|
| GSM8K test | 1319 | Standard; continuity with prior work | Jia & Mu, Phan, Self-Correction Bench, Knowing-Saying (style) |
| GSM-Symbolic P1 / P2 | 600 / 400 | Same problems with +1/+2 reasoning clauses → longer dependency chains; templates kept within one split | Mirzadeh et al. ICLR 2025 |
| MATH-500 (numeric answers) | ≤500 | Harder; tests transfer beyond grade-school arithmetic | Phan et al. (MATH500-Error) |
| Synthetic serial chains | 900 | Depth 3–8, every step on the dependency path; depth and remaining steps controlled by design | Knowing-Saying Gap (synthetic multi-hop) |
| BBH multistep arithmetic (generated in the task's format) | 400 | Highest propagation in prior work (≈64%) | Jia & Mu (BBH) |

## 3. Pilot evidence (look_v1, Qwen2.5-7B-Instruct, 358 problems, 1441 injections)

- FOLLOW vs RECOVER readable at the injection token: test AUROC 0.965; +0.094 [0.046, 0.153] over
  all covariates; stable over 5 splits (gain 0.089 ± 0.009).
- **Small errors are followed ~93% of the time** (+1, digit swap). Only large errors (×10) are
  sometimes caught (41% followed). Within ×10: probe 0.90 vs covariates 0.62, gain +0.27 [0.16, 0.39].
- Value tracking (does the residual stream still carry the true digit?) was null.
- Weak spots fixed in this run: single shuffle control (now 50-permutation null), few catchable
  cases (now three random operators and harder datasets), MATH under-represented (more datasets).

## 4. Pre-registered analysis (fixed before this run)

**Headline subset ("catchable"):** on-path injections with relative error ≥ 50%. Rationale from the
pilot: below that, the model essentially never catches the error, so there is nothing to predict.
P1 on *all* injections is reported alongside.

**H1 (signal):** P1-catchable test AUROC ≥ 0.70 and permutation p < 0.05 (50 shuffles).
**H2 (beyond covariates):** within-problem AUROC ≥ 0.65 and gain over all covariates ≥ 0.05 with the
bootstrap CI above 0. Covariates = difficulty (pass rate, level, dataset), structure (position,
downstream steps, equation count, token count, operator, error size) and surprisal of the injected
digits.
**H3 (useful):** at a 20% intervention budget, probe-gated "Wait" yields higher final accuracy
than surprisal-gated and random-gated "Wait" on held-out steps.
**H4 (causal, exploratory):** steering along the FOLLOW−RECOVER direction moves FOLLOW rate
monotonically with alpha, and random directions of equal norm do not.

| Outcome | What we can say |
|---|---|
| H1+H2+H3 | The model's state at the error token predicts catching beyond surface cues, and a monitor can use it |
| H1+H2, not H3 | Readable but not actionable with "Wait" (report as diagnostic) |
| H1, not H2 | Propagation probes mostly read error size / plausibility (correction to Jia & Mu) |
| H4 null | Diagnostic, not causal (consistent with Jia & Mu and Hidden Error Awareness 2026) |

## 5. Probe list

| ID | Target | Position | Model |
|---|---|---|---|
| P0 | corrupted vs clean (sanity) | last token of the written number | logistic |
| **P1** | **FOLLOW vs RECOVER** | same | logistic, layer sweep |
| P1r | follow rate over greedy + 4 samples | same | ridge |
| P2 | true vs injected last digit, downstream representation | same | 10-way logistic trained on clean runs |
| P3 | silent vs visible recovery | same | logistic |
| P4 | contamination horizon (# later equations using a wrong value) | same | ridge |
| P5 | problem-level follow rate | last prompt token, before any CoT | ridge |
| W | probe-gated Wait policy | — | offline simulation over both continuation arms |
| S | steering along the P1 direction | all positions at the P1 layer | additive, ± random controls |

## 6. Known limits

- One model (Qwen2.5-7B-Instruct). DeepSeek-R1-Distill-Qwen-7B is supported by config; it needs
  longer budgets and is the obvious second run.
- Injections are value edits of verified equations; operator and premise errors are not covered.
- "Wait" is one intervention; a re-generation or tool-call check would be the next comparison.
- Labels rely on the model writing explicit equations; prose-heavy traces contribute fewer sites.
