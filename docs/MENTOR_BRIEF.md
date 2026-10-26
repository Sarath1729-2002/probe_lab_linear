# Mentor brief: what does a frozen LLM know about its own continuation?

**One sentence.** We read, with linear probes only, the quantities a deployment monitor would act on (will this run away, will it obey the constraint, is the answer already settled, how much is left) from the residual stream of Qwen2.5-7B-Instruct, against explicit observable and oracle baselines, with deterministic labels and a pre-declared strict scorecard.

**Why these signals.** Each has a decision attached: cut a runaway generation before the compute is spent, predict an instruction violation before generating (controllability), stop when the answer is already settled (and check the visible chain against the internal state), detect silent failure to deliver an answer. They extend "How Much is Left?" (remaining length is linearly encoded) to quantities with a safety use.

**What is new relative to the length paper.** Runaway / non-termination (the paper discards those generations as right-censored), constraint compliance with exact verifiers, answer settledness and readiness along the solution, and a checkpoint design that does not leak length through position.

**What would count as a result.** SUPPORTED rows under the strict rule (beats the best observable baseline with a CI above zero, R2 >= 0.10, and holds in at least half of the leave-dataset-out tests). Anything else is reported as PARTIAL, WEAK or NOT SUPPORTED; a negative row is still a finding about linear readability.

**Known limits.** One model, greedy labels, observational (no steering yet), small hand-picked constraint set, MATH restricted to integer answers.

**Next.** Second model (R1-distill, same problems), then the steering study on the directions that are SUPPORTED.
