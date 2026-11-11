# organic v6.1 — three checks on the finished v6 run (observation only)

Same items, same stored replies, same run folder. Adds prompts and regenerates only replies that were cut off.
1. No-label controls: `"The answer is X."` alone (m_bare) and `The answer key says: ...` (m_key).
2. Reason first: every source again with "Think it through step by step first." (r_*), plus the bare question.
3. Cut-off replies (used the whole token budget, any family, both models): regenerated with a larger budget;
   the original is kept in each row as text_v0 / outcome_v0e.
Then the v6 analysis + report are redone (some outcomes change) and REPORT_v61.md / COMPARE_v61.md are written.

## Commands (TamIA)

    cd ~/probe_lab_linear
    tar xzf ~/organic_v61_overlay.tgz
    # check the new code is in place (each should print 1 or more)
    grep -c "def stage_extend" organic/run.py
    grep -c "max_tokens_reason" config_organic_v6.yaml
    grep -c "variants61" slurm/organic_v61.sbatch
    # back up the v6 replies once (extend rewrites gen.jsonl in place)
    tar czf $SCRATCH/organic_v6_gen_backup.tgz -C $SCRATCH/organic_runs $(cd $SCRATCH/organic_runs && ls organic_v6/*/*/gen.jsonl)
    cp $SCRATCH/organic_v6_data/variants.jsonl $SCRATCH/organic_v6_data/variants_v6.jsonl
    sbatch slurm/organic_v61.sbatch

No login-node build step and no downloads: the new prompts are appended to the existing variants.jsonl by the job.
Expected: about 29k new prompts per model plus the cut-off replies; under 10 h on 4 H100s.

## Progress / completion

    squeue -u $USER
    tail -f logs/ep_organic_v61-<jobid>.out          # ends with "ALL DONE"
    grep -h "\[extend\]" logs/org61-<jobid>-*-extend-*.log

## Outputs

    reports/organic_v6/COMPARE_v61.md                  # the main v6.1 table, both models
    reports/organic_v6/fig_reason_first.png
    reports/organic_v6/<model>/REPORT_v61.md
    reports/organic_v6/<model>/REPORT.md, reports/organic_v6/COMPARE.md   # v6 report, redone with the extended replies

Send back:

    tar czf ~/organic_v61.tgz --exclude='*.npy' -C ~/probe_lab_linear reports/organic_v6 logs -C $SCRATCH/organic_runs organic_v6
