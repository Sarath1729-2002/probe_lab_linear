# probe_lab

Linear (ridge) probes with **regression labels** on **Qwen2.5-7B-Instruct / GSM8K**. Three experiments, one run, all on frozen
prompt-end hidden states (every layer), all labels deterministic or measured by sampling.

| | question | label (regression) | bar the probe must beat |
|---|---|---|---|
| **E1 knowledge boundary** | from the question alone, does the residual stream encode how likely the model is to solve it? | `p_clean` = fraction of 16 sampled answers correct; `ent_clean` = answer entropy over the 16 (semantic entropy for numeric answers); `lpg_clean` = log P(gold) with no reasoning | question length/numbers, bag-of-words, and an **oracle** that sees the reference-solution step count |
| **E2 hint arbitration** | with a planted *wrong* answer (answer-key framing / "I worked it out" framing), is *whether the model will follow it* readable before it writes anything? | `follow_*` = fraction of 16 samples ending in the planted answer; `gap_*` = log P(hint) - log P(gold) forced, no CoT; residual versions with clean difficulty regressed out | all of the above **plus oracle clean pass-rate** (does the probe read more than difficulty?); cross-framing transfer auth -> user |
| **E3 injected-concept readout** | a concept direction is added to the residual stream; can a probe read injection/strength better than the model's own "do you notice an injected thought?" answer? | strength `s` (known), injected (0/1); held-out **concepts** | verbal self-report AUROC; causal null: layers below the injection layer must be ~0.5 |

Splits: by problem (70/10/20) and a **shift** regime (train on <=3-step problems, test on >=5-step). E3 splits by concept.
Probe = per-layer closed-form ridge, lambda by validation MSE, layer by validation R2, test touched once; paired bootstrap CIs.
`PASS` in the report = CI lower bound of (probe - best baseline) > 0 and probe R2 > 0.05. Nothing is tuned on test.

## Run (TamIA)
```bash
bash cluster/setup.sh                 # login node: venv, model + GSM8K download, problems.jsonl
MODE=gate bash cluster/submit.sh      # 240 problems, every stage, report in ${RUN_DIR}-gate/analysis/report.md  (<1h)
MODE=full bash cluster/submit.sh      # ~4000 problems: states -> sample -> introspect -> analyze (chained, resumable)
MODE=analyze bash cluster/submit.sh   # re-run only the analysis
```
Knobs (env): `N_SAMPLES=16 MAX_NEW=512 EXTRA_TRAIN=2700 MAX_SEQS=192 SBATCH_ACCOUNT=aip-oberman`. Outputs: `$RUN_DIR/{samples,states,introspect.npz,analysis/report.md}`.

## Tests
`pytest tests` (tiny random Qwen2; checks prompt-end state equals a direct forward, resumability, injection causality, probe recovery).

---
# Main study (v3): pre-generation readout of hint adoption

**Question.** Does the residual stream at the last prompt token, before any output, linearly encode how likely Qwen2.5-7B-Instruct is to adopt a
*planted wrong answer*, beyond difficulty and beyond what the prompt text reveals?  Seven graded framings of the same wrong hint
(`auth user weak peer teacher grader pressure`), 16 sampled solutions per (problem, framing), ~4000 GSM8K problems.

**Metrics (declared up front)** M1 R2/Spearman of the val-selected best-layer ridge probe; M2 delta-R2 vs the best baseline (paired bootstrap 95% CI; PASS = CI>0 and R2>0.05, on
both the iid and the step-shift split); M3 AUROC for "adopts" (rate>=0.25); M4 leave-one-framing-out R2; M5 steering dose-response vs random directions.
Full layer sweep (all 28 layers, CI band), baseline ladder (question features, bag of words, step-count oracle, measured-difficulty oracle, hint-number features, early-layer states,
other-framing behaviour oracle), label noise ceilings, permutation control.

```bash
MODE=v3gate bash cluster/submit.sh   # dry run on 240 problems (all 8 conditions, whole chain)  -> ${RUN_DIR}-gate/analysis_v3/REPORT.md
MODE=v3 bash cluster/submit.sh       # the study: reuses clean/auth/user from the first run, samples the 5 new framings (~5-6 h), analysis, steering, REPORT.md
```
Output: `$RUN_DIR/analysis_v3/{REPORT.md, fig*.png, *.csv, directions.npz}`.  Stages: `states -> sample -> analyze3 (compute) -> steer -> report3`.
