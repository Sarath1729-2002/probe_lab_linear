# Deference to pressure in language models

Code for a study of what happens inside a language model when it gives up an answer it knows because something in
the context pushes it: a user's guess, an authority, a misleading document, pushback, an incentive, or an instruction
to lie.

The question is simple. After the model gives in, can you still read from its internal state the answer it would have
given without the pressure? And can a monitor use that when asking the model itself does not work?

## Setup

- Items: MMLU, ARC, CommonsenseQA and OpenBookQA, multiple choice, split 70/30 into train and test
  (1,350 test items).
- The model's answer with no pressure defines what it knows.
- Ten pressure types: user suggestion, authority, pushback, "are you sure?", a misleading answer key, few-shot examples
  with wrong labels, an incentive, an instruction to lie, a deceptive persona, sandbagging. Plus a 17-type
  peer-pressure set and a set of 48 everyday phrasings.
- Labels come from answer letters. No LLM judge anywhere.
- Models: Qwen2.5-3B/7B/14B, Qwen3-8B, Llama-3.1-8B, Mistral-7B (instruct versions).

After each reply the model is asked "Is (X) the correct answer?" and the last-token state is read. The main probe
maps that pressured state to the model's own Yes/No verdict on the same item in the conversation without pressure.
It is trained leave-one-pressure-type-out and needs no dishonesty or truth labels.

## Results so far (six models)

| | range over models |
|---|---|
| Detection of pressured departures, pooled AUROC | 0.76 – 0.91 |
| Gap to a probe supervised with departure labels (3 held-out models) | -0.03 to 0.00 |
| Gain over asking the model | +0.02 to +0.21 |
| Gain over an instructed-lie probe | +0.09 to +0.30 |
| Asking the model, under authority / misleading answer key | 0.17 – 0.73 |
| Instructed lies the model disowns when asked | 97 – 100 % |
| Harmful vs helpful answer change: unpressured confidence / probe | 0.65 – 0.82 / 0.50 – 0.73 |

All gains have bootstrap 95% CIs over items above zero.

What does not hold:

- The probe is not a general belief readout. Trained without the social pressure types, it largely fails on them
  (0.41 – 0.76).
- A pre-registered margin test, fixed before scoring the last three models (`FROZEN_DETECTOR.md`, commit `83d4332`),
  failed on Llama-3.1-8B (0.76 against 0.80). On that model the Yes/No readout disagrees with the model's own
  generated answer half the time.
- Recall at a strict alarm budget is modest: 6 – 13 % at 1 % false positives.

Short version: being unendorsed by the unpressured model is readable without labels and beats self-report, but telling
harmful deference from legitimate updating needs the model's own knowledge on top. Prompts and steering stay within
that limit. Fine-tuning on balanced pressure data goes past it in 5 of 5 models (`organic/wall.py`).

## Layout

```
organic/
  selective.py     items, bare answers, pressure prompts, steering and evaluation (shared base)
  sel_more.py      causal layer choice, LoRA data-mix runs
  honesty.py       lie direction vs caving direction
  knowsay.py       main study: generation, follow-up readout, probes, detectors, steering, reports
  ks_review.py     stratified analyses, rerun oracle, cluster hold-out, baselines (CPU)
  ks_controls.py   copy control, paired bootstrap, honest-mistake check (CPU)
  wall.py          self-knowledge wall: can any method beat the model's own confidence? (CPU)
config_sel_v2.yaml   models, datasets, run blocks (knowsay, knowsay_peer, knowsay_nat)
slurm/               job scripts (final.sbatch runs the full study, resumable)
tests/               CPU smoke tests on a tiny random model
FROZEN_DETECTOR.md   detector definition fixed before the confirmation models
```

Older files in `organic/` (`kill.py`, `report_v*.py`, ...) belong to an earlier error-propagation study and are kept
for reference.

## Running

Python 3.12 with `torch`, `transformers`, `peft`, `datasets`, `scikit-learn`, `scipy`, `numpy`, `pyyaml`.

```
# full study on a 4-GPU node (every stage skips if its output exists)
sbatch slurm/final.sbatch

# one model, one stage at a time
python -m organic.knowsay gen follow fit steer report --config config_sel_v2.yaml --model qwen7b
python -m organic.knowsay gen follow fit report --config config_sel_v2.yaml --model qwen7b --run knowsay_peer

# analyses on finished runs (CPU)
python -m organic.ks_review   --config config_sel_v2.yaml --models qwen3b qwen7b mistral7b llama8b qwen3_8b qwen14b
python -m organic.ks_controls --config config_sel_v2.yaml
python -m organic.wall        --config config_sel_v2.yaml

# smoke test
ERRPROP_DEVICE=cpu python tests/smoke_knowsay.py /path/to/tiny_model
```

Paths for data and runs are set at the top of `config_sel_v2.yaml` (`data_dir`, `out_dir`). Reports go to
`reports/sel_v2/`. Everything is deterministic given the seed.

## Status

Work in progress. Next: frontier models through their APIs (black-box part only), free-form answers, a 32B model,
and chain-of-thought hints.

Sarath Krishna Chingapurathu, Mila / Université de Montréal
