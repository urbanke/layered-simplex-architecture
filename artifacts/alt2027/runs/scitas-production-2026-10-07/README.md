# Queued ALT production handoff — 7 October 2026

Slurm job **40159544** is queued behind the remaining depth validation
(job 40152959) and throughput pilot array (40158582). It runs the explicit
numerical-admission and scheduling checks when those jobs finish. Only passed
checks can submit the production array; failed or incomplete evidence is retained
with its status. The `afterany` dependency allows the gate to record failures as
well as successes. Production has not started at this recorded milestone.

The complete frozen plan has **3,904 disjoint jobs**, retaining all nine experiment
families, 1,000 primary/power/spectrum/depth-curve repetitions and 5,000 factorial
profiles per cell. It uses 20-trial work blocks, 500 for factorial, and numerical
batches of 20. Its canonical SHA-256 is
`d5a337662543663845d6b2ead796a14c05d5589e5f5930eab9c2315d4b98d17f`.
The compressed JSON is the exact submitted plan; all jobs were independently
reconstructed and matched before queueing. Scientific source is the clean
`9ec82bcef549ad593dee5450e974d83de44ddc77` checkout.

The corrected full power suite passed on both the laptop and SCITAS. Final Linux
regressions passed all 388 tests and the full Appendix C checks. Their verified
archives are under the corresponding validation milestones. The handoff evidence
index refers to the complete immutable SCITAS Runs. Exact launchers, scheduling
rules, configuration hashes and the actual submission record are preserved here.

The resource gate chooses 4, 8 or 16 independent nodes, each with 72 workers, 440 GiB
and a 24-hour limit. It verifies representative timing Runs and unchanged settings,
uses twice the measured costs plus 60 seconds per job and a one-hour controller
reserve, and checks the actual deterministic controller assignments. It refuses
an unassessed resource configuration. These are scheduling projections from
recorded profiles. Model grids, sampling counts and numerical tolerances are fixed.

After admission, the handoff saves the production array ID under
`/scratch/urbanke/lsa-alt2027/runs/production-launch-9ec82bc-001/submitted.json`.
The production root is `.../runs/production-9ec82bc-001`. Controller numerical
failure cancels sibling controllers; completed jobs and failed attempts remain
available. Scheduler termination is separately visible in Slurm records.

A postprocessing job is then queued after successful completion of the entire
array. It verifies and merges all 3,904 jobs, regenerates the existing figures and
tables, and verifies all nine merged Runs plus the report Run. It writes separate
immutable logs/outcomes and asset hashes. Its job ID is recorded in
`postprocess-submitted.json` beside the production submission record. Generated
assets remain in scratch for inspection and durable archiving before manuscript
integration. Overleaf is unchanged.

The eight exact handoff inputs are pinned by `handoff-inputs-9ec82bc.json` and
checked again before launch. The code avoids duplicate submissions by refusing
existing launch/output records, and preserves the scheduler output for any
unknown submission outcome. The mock cancellation check uses a fake scheduler
and verifies propagation of a controller failure status without cancelling jobs.
