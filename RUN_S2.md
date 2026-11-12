# Steering v2 (organic/steer2.py) — tonight's run

Reuses the v6 activations and replies. Directions are learned on half the items; steering is tested on the other half.
Per model: a short calibration pass (~280 replies), then 7 conditions x ~4,200 prompts (~600 held-out known items x
bare question, user and document wrong/right, user 2nd turn, "Are you sure?").

    cd ~/probe_lab_linear
    tar xzf ~/organic_s2_full.tgz
    python -c "import organic.steer2; print('ok')"
    sbatch slurm/s2.sbatch

Progress: `squeue -u $USER`; `tail -f logs/ep_s2-<jobid>.out` (ends with ALL DONE); `ls logs/s2-<jobid>-*`.
Outputs: reports/organic_v6/COMPARE_s2.md, reports/organic_v6/fig_s2_tradeoff.png, reports/organic_v6/<model>/REPORT_s2.md
Send back:

    tar czf ~/s2.tgz -C ~/probe_lab_linear reports/organic_v6 $(ls logs/ep_s2-* logs/s2-* 2>/dev/null) -C $SCRATCH/organic_runs organic_v6/qwen7b/s2 organic_v6/llama8b/s2
