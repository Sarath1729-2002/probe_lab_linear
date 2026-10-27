# Direction: Does the model know it will follow an error?

**One question.** When a wrong value enters a model's chain of thought (its own slip, a bad tool
output, a corrupted sub-agent trace), the model then does one of three things:

- **FOLLOW**: carries the wrong value through to a wrong final answer,
- **RECOVER**: reaches the right answer anyway, either
  - *visibly* (says "wait", or re-writes the true value), or
  - *silently* (the trace never repairs it, but the answer is right),
- **OTHER**: goes wrong some other way.

We ask whether a **linear probe on the residual stream at the moment the error is written** can
tell which of these will happen, and whether that signal survives once problem difficulty is held
fixed.

## Why this is the gap

| Paper | What it found | What it leaves open |
|---|---|---|
| Jia & Mu 2026 (2609.25366) | Propagation is decodable at the injection point (linear AUROC ≈ 0.78, Gemma-2-9B) | Their own analysis attributes 98.8% of propagation deviance to task difficulty. The probe is never difficulty-controlled. Three-way labels rely on an LLM judge. |
| Knowing-Saying Gap 2026 (2608.07528) | Corruption is detected (AUROC > 0.98) but failure prediction is at chance | Weak models (23–31% clean accuracy), synthetic templates only |
| Phan et al. 2026 (2603.16331) | Recovered vs unrecovered separable at *answer* tokens (0.88–0.99), critique vector steers | Read-out happens after the model has already chosen; errors are GPT-5-written and xVerify-graded |

Three papers, three answers. **Our contribution is the controlled version**: on-policy traces,
deterministic injections, a symbolic FOLLOW label (no judge), several injections per problem so
the probe must explain *within-problem* variation, and covariate baselines that include the
model's own token surprisal.

## Safety framing (the monitor decision)

CoT monitoring assumes the written chain constrains the answer. Errors that enter the chain
(tool outputs, retrieved text, sub-agent traces, adversarial edits) break that assumption in two
opposite ways:

1. **FOLLOW**: the monitor must act *before* the wrong value is used. A probe that fires at the
   injection point buys that time.
2. **Silent RECOVER**: the written chain is wrong but the answer is right, so the CoT is not
   what drove the answer. A monitor reading that chain is reading decoration.

So the probe gates a concrete decision: *for this piece of content that just entered the reasoning,
should we (a) intervene now, (b) trust the CoT, or (c) distrust the CoT and look elsewhere?*

## The probes (all linear, all labels exact)

| ID | Probe | Label | Position | Model |
|---|---|---|---|---|
| P0 | Corruption detection (sanity, known result) | clean vs injected | last token of the injected number | logistic |
| **P1** | **FOLLOW vs RECOVER (main)** | final answer = propagated answer vs = gold | same | logistic |
| P1r | Follow rate | fraction of k+1 continuations that FOLLOW | same | ridge |
| **P2** | **Value tracking** | last digit of the *true* vs *injected* value | same | multinomial logistic, trained on clean runs |
| P3 | Silent vs visible recovery | among RECOVER: no correction marker and true value never re-written | same | logistic |
| P4 | Contamination horizon | # later equations that use the wrong value | same | ridge |
| P5 | Prompt-time susceptibility | problem-level follow rate | last prompt token, before any CoT | ridge |

P1 + P2 + P3 are the paper. P4/P5 come free from the same runs.

## Hypotheses and decision rule (fixed before running)

- **H1 (signal exists).** P1 test AUROC ≥ 0.70 at the dev-selected layer.
- **H2 (it is not just difficulty).** P1 *within-problem* AUROC ≥ 0.65 on test, and
  [covariates + probe] beats [covariates alone] by ≥ 0.05 AUROC with a bootstrap CI above 0.
- **H3 (value tracking).** In mid layers, P2 decodes the *true* digit on corrupted RECOVER runs
  more often than on FOLLOW runs.

| Outcome | What we write |
|---|---|
| H1 + H2 hold | "Models represent whether they will absorb an error, beyond difficulty and surprisal" (main result) |
| H1 holds, H2 fails | "Published propagation probes are difficulty probes" (a correction to Jia & Mu, still publishable) |
| H1 fails | Negative result; report label distributions and stop this branch |

## What this run is (and isn't)

A **look-through run**, not the final paper: Qwen2.5-7B-Instruct, ~500 GSM8K + ~250 MATH-500
problems, up to 6 on-path + 1 off-path injections each, greedy + 4 sampled continuations. Big
enough to see whether the label distribution supports the question and whether P1/P2 move at all.
R1-Distill-Qwen-7B is supported as a second model once this looks alive.

## Known risks

- **Few FOLLOW cases on GSM8K.** Prior work saw 4–12% propagation on GSM8K. Our cut is mid-line
  (right after the wrong number), which should raise uptake, and MATH adds harder cases. The
  feasibility gate (50 problems) checks this before the full run.
- **Few mixed-outcome problems** limits within-problem AUROC. The report prints how many there are.
- **OTHER**: the model may take a different route that uses the wrong value but lands on a third
  number. We report the OTHER rate and an "uptake" flag (injected number re-appears).
- **Surface leakage**: the injected digit is literally in the text. P0 will be ~1.0 trivially;
  that is why P1 compares FOLLOW vs RECOVER *among corrupted runs only*, and why the off-path
  control exists (corrupted but cannot propagate).
