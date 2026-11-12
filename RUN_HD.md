# Deference and honesty (organic/honest_defer.py) — tonight's experiment

Reuses the v6 activations and replies; adds ~2,000 short forward passes (honest vs deceptive system prompt) and
steered generation on 150 known items per dataset (6 conditions x ~4,500 prompts per model).

    cd ~/probe_lab_linear
    tar xzf ~/organic_hd_full.tgz                      # whole organic/ package + config + slurm/hd.sbatch
    python -c "import organic.honest_defer; print('ok')"
    sbatch --dependency=afterany:687772 slurm/hd.sbatch    # starts after the v6.1 job (687772) ends

Progress: `squeue -u $USER`, `tail -f logs/ep_hd-<jobid>.out` (ends with ALL DONE), `ls logs/hd-<jobid>-*`.
Outputs: reports/organic_v6/COMPARE_hd.md, reports/organic_v6/<model>/REPORT_hd.md and fig_hd.png.
Send back:

    tar czf ~/hd.tgz -C ~/probe_lab_linear reports/organic_v6 -C $SCRATCH/organic_runs organic_v6/qwen7b/hd organic_v6/llama8b/hd
