# Uncertainty probing - proof of concept

**Question.** Can a *linear* probe on frozen hidden states predict how uncertain the model's final answer is, with an exact regression label, on exact-answer reasoning tasks - and does that survive baselines that need no probe?

Closest published work: Semantic Entropy Probes (Kossen et al. 2024; hidden states -> semantic entropy), Reasoning Theater (Boppana et al. 2026; attention probes decode the final answer along the trace on multiple choice, linear probes near chance). Not found: linear probes with continuous answer-distribution labels on exact-answer reasoning, read at the prompt end and along the trace. This POC tests whether that is worth a full study. It is a search of limited depth, not a novelty claim.

## What it runs (re-uses the finished main run: problems, greedy generations, prompt-end states, trace checkpoints)

* subset: `UNC_N` problems per dataset (gsm8k, math), deterministic by hash, nested (n=40 is inside n=400).
* **prompt-end rollouts**: `N_ROLL`=16 samples (T=0.7, top-p 0.95) of the plain prompt -> labels `pass_rate`, `ans_ent` (entropy of the empirical final-answer distribution, nats), `maj_frac` (share of the most common answer; needs no gold).
* **checkpoint continuations**: at the model's own greedy prefix of 48 and 96 tokens (the trace checkpoints o48, o96), `N_CONT`=8 sampled continuations -> `c_pass`, `c_ent`, `c_maj`: *is the model still uncertain after 48 tokens?*
* **free signals**: teacher-forced mean next-token entropy / NLL of the greedy solution (all tokens, first 48, first 96) - the cheap uncertainty signal a probe has to beat.
* probes/baselines/splits are the main study's (ridge per layer, lambda on validation, layer on validation R2, test touched once, paired bootstrap over problems; splits iid / easy->hard shift / leave-dataset-out).

Cost guess (not timed): ~6-9M generated tokens; 4 H100 for roughly 0.5-1.5 h with HF generate. `UNC_BUDGET_H` (default 3) stops cleanly and `MODE=unc` resumes.

```bash
cd ~/model_introspection_probe_suite      # after the usual paste / git pull, main run finished
UNC_N=40 MODE=unc bash cluster/submit.sh  # optional smoke run (2 datasets x 40 problems), output in $RUN_DIR/unc_n40
MODE=unc bash cluster/submit.sh           # the POC: UNC_N=400 -> $RUN_DIR/unc_n400/analysis/UNC_REPORT.md
```

## Decision rule (fixed before any real result)

Noise first: the labels come from finite samples. Split-half reliability (even vs odd rollouts, Spearman-Brown) is the **ceiling** for probe R2; every result is shown next to it.

Positive controls (pipeline validated only if they reproduce): gold-answer-size probe pooled iid R2 >= 0.15; Spearman(pass_rate label, greedy correct) >= 0.40; remaining-length probe at o48 R2 >= 0.30. If they fail the report says **CONTROLS FAILED** and nothing else is read.

Per label and scope:

| status | condition |
|---|---|
| SUPPORTED | beats the best *observable* baseline (CI lower bound > 0 and R2 > 0.05) in iid **and** in >= 2/3 of {shift, leave-dataset-out} |
| PARTIAL | beats it in iid only |
| NEGATIVE | label reliability >= 0.6, controls pass, and the CI upper bound of the gain over observable is < 0.05 (the gain is ruled out) |
| INCONCLUSIVE | anything else - the `why` column says whether it was label noise, controls, or an interval too wide |

Decision: **GO** if a primary label (`ans_ent`, `pass_rate`) is SUPPORTED at the prompt end (or a checkpoint label is: "trace-time GO"); **LEAD** if something is PARTIAL; **STOP** if every primary label is NEGATIVE; otherwise **INCONCLUSIVE** (fix = more rollouts / problems, not a conclusion).

Observable baselines: dataset id, question features, bag of words, early layers H4/H8; at checkpoints also the visible prefix text and the prefix token entropy. Oracle baselines (greedy-run summaries with token entropy and correctness; prompt-end rollout statistics at checkpoints) are context only.

## Known limits (state them in any writeup)

* two datasets: leave-dataset-out is train-on-one / test-on-the-other. A SUPPORTED label is a lead, not a general claim.
* entropy is not monotone in a latent difficulty (it peaks at pass rate 0.5); a linear probe reads it through whatever monotone part the realised data has. `pass_rate` and `maj_frac` are the cleaner targets; `ans_ent` is the one that links to semantic entropy.
* plug-in entropy from 16 (or 8) samples is biased low; the label is a noisy proxy, hence the ceiling.
* many easy problems are unanimous (label saturation); the report lists the share of mixed problems.
* the group's semantic-entropy probe is not included as a baseline here; the exact-answer entropy is its closed-form analogue for integer answers.
