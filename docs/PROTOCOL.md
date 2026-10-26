# Protocol (pre-declared; the scorecard in REPORT.md applies it mechanically)

## Question
What does a frozen LLM's residual stream **linearly** encode about its own continuation, in the quantities an operator or monitor would act on?
All labels are deterministic functions of greedy-decoded text (no annotators, no judge model, no sampling noise in the main labels).

## Signals (the "decision" column is why each one is in the suite)
| id | label (see `table.py`) | read at | decision it would support |
|---|---|---|---|
| finish | `runaway` (greedy hit the token cap), `runaway_rate` (share of 4 samples that did) | plain prompt end | cut / reroute before the compute is spent; overthinking and loop risk |
| length | `logT` (finished solutions) | plain prompt end | scheduling and budgeting; replication of "How Much is Left?" |
| spread | `len_sd` (std of log-length over its own samples) | plain prompt end | does the model know how uncertain its own length is |
| deliver | `noans` (no parsable final answer) | plain prompt end | catch silent failure to deliver |
| correct | `correct` | plain prompt end | reference point (a known probe target) |
| answer_size | `ans_logmag`, `ans_ndig` of the GOLD answer | plain prompt end | does the model estimate the result before reasoning |
| comply | `comply` (deterministic IFEval-style check), `over` (log words/limit), `c_runaway` | constrained prompt end | predict an instruction violation before generating: controllability |
| settled | `settled_p` = P(own final answer) if the answer is forced now | state inside the solution | early stopping; whether the visible chain matches the internal state |
| ready | `ready_p` = P(gold answer) if forced now | state inside the solution | answer-readiness (overthinking: reasoning that continues after the answer is known) |
| remaining | `log_rem` = log(1 + tokens still to come) | state inside the solution | replication of the length-progress signal, with a fair baseline |

Constraints (one per problem, assigned by a seeded hash; a truncated response never complies): `maxw50`, `maxw100`, `maxw200` (word limits), `nocomma`, `endphrase`.

Checkpoints are **absolute token offsets** (0, 24, 48, 96, 192) plus `pre` (just before the final `\boxed`). A fixed *fraction* of the solution would make position reveal the total length and trivialise `remaining`; offsets are valid only if at least 8 tokens remain; `pre` is not used for `remaining`.

## Probe and baselines
Closed-form ridge per layer on frozen states (standardised features, lambda by validation MSE, layer by validation R2, test touched once, paired bootstrap CIs over test problems). Binary labels are regressed as 0/1 and additionally reported as AUROC.

Baseline ladder (same ridge code). **Observable** = computable from what exists at that point: dataset identity (`Bds`), question text features (`Bq`), hashed bag of words (`Bbow`), constraint identity and limit (`Bcon`), visible solution-so-far (`Btext`), the model's own early layers (`H4`, `H8`). **Oracle** = also sees what the model cannot know yet: design difficulty (`Bd1`), behaviour under the other prompt (`Bother`, `Bplain`: the model's natural unconstrained length, correctness, ...), whether the answer string is already in the text and the total length (`Btext+answer-mentions`).

## Regimes
`iid` (stored 70/10/20 split by problem), `shift` (per dataset: train on the easiest 35% by design difficulty, test on the hardest 35%), `lodo:X` (leave-dataset-out: fit on the other datasets, test on all of X).

## Decision rules (strict)
* verdict PASS = CI lower bound of (probe R2 - best baseline R2) > 0 **and** probe R2 > 0.05. Reported against the observable ladder (`verdict_obs`) and the oracle ladder (`verdict_oracle`).
* **SUPPORTED** = pooled `iid` PASS vs observable, probe R2 >= 0.10, **and** the same PASS in at least half of the leave-dataset-out tests.
* PARTIAL = passes `iid` but not the generalisation requirement. WEAK = beats the baseline but R2 < 0.10. NOT SUPPORTED otherwise.
* Checkpoint signals are judged at the 48-token checkpoint, and the number of the 6 checkpoints that beat the text baseline is reported next to it.
* "Beyond oracle" (`verdict_oracle` = PASS) is a separate, stronger claim and is never inferred from the observable verdict.
* Silent settledness: restricted to checkpoints whose visible text does not yet contain the model's final answer, the probe is compared to the text baseline by AUROC (n and positives reported; read small n with care).

## What a result would and would not mean
* A SUPPORTED row says the quantity is linearly readable and more than what is visible in the text and the model's early layers; it does not say the model uses that direction (that needs the deferred steering study).
* NOT SUPPORTED is evidence about this model, these datasets and this readout, not about absence in larger models.
* Everything is one model (Qwen2.5-7B-Instruct) and greedy decoding for the labels. The first replication is a second model (`MODEL_ID`).
