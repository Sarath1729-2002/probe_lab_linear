"""Observational study: wrong information in the prompt, nothing edited in the model or its output.

One model (Qwen2.5-7B-Instruct). Existing datasets, grouped into buckets; the wrong information comes
from the datasets themselves (their incorrect answers, distractors, swapped passages, other models' wrong
solutions) or, where a dataset has none, another item's correct answer (as NQ-Swap does).
One manipulation for every bucket: Sharma et al. (2023)'s answer-sycophancy templates.

Pipeline:  sources (login) -> build -> gen -> extract -> fit -> traj -> report
"""
