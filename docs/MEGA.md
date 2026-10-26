# Mega run - GSM8K only, one overnight job (Qwen2.5-7B-Instruct, 4xH100)

Four independent threads. A failure or a null in one never blocks the others. Every thread has a decision rule fixed HERE, before the data exists. Output: one report, `$RUN_DIR/mega/analysis/MEGA_REPORT.md`, whose first section ("Where to pull") lists what to pursue.

## Corrected protocol (applies to every thread)
The uncertainty POC picked (layer, lambda) on a ~80-row validation split; it landed on layer 2 for three labels (R2 0.06-0.22) while per-dataset fits at layers 20-28 scored 0.11-0.34. That is selection noise, not absence of signal.
Here: (layer, lambda) by **grouped 5-fold CV over the whole training pool** (folds never split a problem), refit on the pool, test touched once. **Bootstrap over problems** (rows from one problem are never split). Two regimes: **iid** (fit+val -> test) and **shift** (easiest 35% by design difficulty -> hardest 35%). Baselines go through the identical code; the verdict is against the best **observable** baseline (names with "(oracle)" are shown but never decide).
Generic status: SUPPORTED = PASS (CI lower bound of the gain > 0 and probe R2 > 0.05) in iid AND shift. PARTIAL = one regime. NEGATIVE = gain CI upper bound < 0.05 in every regime. INCONCLUSIVE otherwise, or degenerate label.

## Thread 1 - causal step importance (the new idea)
For each numeric step of the model's own greedy GSM8K solution (<= 8 per solution) we intervene and let the model continue greedily:
`n1` result c -> c+1, `n2` c -> c+max(10, c//10), `del` the step line is deleted, `null` the prefix is regenerated UNEDITED (noise floor). `flip` = final answer differs from the original (unparsable counts as a flip).
Labels: `load` = mean(flip_n1, flip_n2, flip_del); `flip_num` = mean(n1, n2); `flip_del`. Ground truth is an intervention outcome, not a judge's opinion.
Probe input: hidden state at the LAST TOKEN of the step, from one forward over the original solution (causal: later text does not exist for the state).
Baselines: position, step-text features (length, operators, number size, number-in-question), question features, prefix features, early layers (H4, H8); oracle: how often the result is reused later, steps remaining.
Decision: generic status. Degenerate if label std < 0.1 or the null noise floor > 0.15.
Why it might be worth it: a monitor that knows *which step of a chain is load-bearing before the chain continues* is a reasoning-area signal tied to a monitoring decision (where to audit / resample). Adjacent literature (Reasoning Theater, 2603.05488; "From Decorative to Load-Bearing") exists - novelty NOT verified; search before claiming.

## Thread 2 - uncertainty, confirmation on UNSEEN data
2000 GSM8K problems not in the POC (`skip=400`), 16 rollouts each, prompt-end only. Labels: pass rate, answer entropy, majority share. Corrected protocol. Status through `unc_study.status` with its controls (answer-size probe R2 >= 0.15, label corr with greedy correctness >= 0.4). The POC found shift-only gain +0.11..0.13 R2 and nothing at checkpoints; this thread asks whether that survives a clean protocol and 5x more data. Only 21% of GSM8K problems are "mixed", so expect a ceiling.

## Thread 3 - is the length direction causal?
Probe direction for logT (prompt-end, finished GSM8K solutions; layer by CV; the last hidden state is excluded because a decoder-layer hook cannot write after the final norm). Controls: mean-difference direction and 8 random norm-matched directions. Add `alpha*s*unit` (s = std of the probe projection, alpha in {-4,-2,-1,1,2,4}) to the residual stream at that layer at all positions, greedy-decode the GSM8K test problems.
Slope of per-problem log tokens on alpha, bootstrap over problems. **SUPPORTED** if probe slope CI > 0, slope > 1.5x the largest |random slope|, and accuracy changes <= 0.05 for |alpha| <= 2. PARTIAL = moves length but damages accuracy. NEGATIVE = slope CI inside +-0.02. n/a below 30 problems.
Caveat: accuracy noise between conditions can by itself exceed 0.05 at small n; PARTIAL then means "check the table", not "dead".

## Thread 4 - GSM8K-only re-analysis (corrected protocol)
logT, len_sd, correct, ans_logmag (now with an arithmetic-heuristic baseline from the question's numbers), runaway; constrained: c_logT, over, comply; remaining length log_rem at o24/o48/o96/o192. Generic status each. Purpose: separate "signal is real" from "POC protocol was noisy" for everything already on the ledger.

## Timeline (whole nodes, 4 jobs run concurrently)
steps 4 h budget, unc 3 h, steer-prep 45 min -> steer-run 2 h, analysis starts after all (afterany) finish; total wall ~ 4.5-5 h worst case. All stages resumable; budgets cut cleanly. Rerun `MODE=mega` to continue, `MODE=mega-analyze` to redo the report.

## Run
```
MODE=mega-smoke bash cluster/submit.sh   # optional, ~40 min, tiny budgets -> $RUN_DIR/mega_smoke
MODE=mega       bash cluster/submit.sh   # the real thing
cat $RUN_DIR/mega/analysis/MEGA_REPORT.md
```

## Honest expectations
No thread is guaranteed. Most likely: thread 4 confirms length/remaining-length (already known), thread 3 is a clear yes/no, thread 1 either gives a defining new signal or a clean negative. A clean negative with its noise floor and baselines is still reportable. Do not describe anything as SUPPORTED that the report does not.
