# Queued ALT production handoff with sample-size shards — 7 October 2026

**Production has not started at this recorded milestone.** Gate job **40162230**
was submitted at 2026-10-07T20:01:25.313111+00:00. Its `afterany` dependencies
are depth refinement **40160952**, sample-size timing array **40160238**,
spectrum timing **40158600** (array element `40158582_2`), and final Linux
regressions **40162027**. The gate records incomplete or failed evidence and
submits production only after both numerical admission and scheduling pass.
This is an immutable submission snapshot, not a live status report.

The frozen scientific source is `0399cb28b973e5550fcbcbce082e8d36c95c2e1e`. The exact production plan
contains **7,754 jobs**: **4,400 primary benchmark jobs** and **3,354 other jobs**.
Primary work is divided by target, sample size and 20-trial block. The full
original `sampling_n_values` grid, target seeds, global trial IDs, paired samples,
scientific settings and trial totals are preserved. Factorial blocks remain 500
and numerical batches remain 20. There are nine experiment families.

The exact submitted JSON is stored as `production-plan-0399cb2-nsplit.json.gz`,
using an empty gzip filename and `mtime=0`. Its canonical plan SHA-256 is
`0a12bbad6948e427319319f5d74a494b7bb67aa1e8f455d0f1eeee159b62eae6`. The uncompressed file SHA-256 is
`e76da0de8e03b7441c2492fd97046b7e5dee074775749692c7bac58ccf88c272`. Decompression reproduces the
original file byte for byte. The incremental source bundle requires commit
`9ec82bcef549ad593dee5450e974d83de44ddc77`; `git bundle verify` passed.

The scheduling gate chooses **4, 8 or 16 nodes**, each with **72 workers**, 440 GiB
and a 24-hour allocation. It uses all eight heavy sample-size pilots, five other
representative pilots, corrected full power evidence and late-offset replay
records. It checks original and selected sample hashes, runtime/source identity,
measured memory and timing, and the actual controller assignments. Its budget is
twice measured cost plus 60 seconds per job and one hour per controller, including
a conservative load/72 plus longest-job check. These are finite-profile scheduling
projections, not universal runtime bounds.

The archived actual Linux assessment is **pending**, with no selected node count;
the spectrum completion record was missing at its 19:59 UTC snapshot. Nine
synthetic gate checks passed, including node choices, invalid identity/runtime,
and changed sample rejection. Synthetic checks are not numerical admission.
The complete scheduling snapshot, including the exact fixture plan, pilot drivers,
checks and pending assessment, is in `scheduling-evidence/gate-and-pilot-drivers.tar.gz`
with its preserved per-file manifest. The separate
[scheduling validation milestone](../../validation/2026-10-07-scheduling-nsplit/README.txt)
provides the same evidence context. No pending status is promoted to passed here.

The earlier gate **40159544** was cancelled before execution (`00:00:00` elapsed);
its raw records remain in the [previous milestone](../scitas-production-2026-10-07/README.md).
A measured 20-trial heavy cohort and structural timing estimates motivated the
new execution partition. Those estimates are archived as estimates. Depth job
**40152959** was separately superseded by **40160952**, using a new 24-hour
allocation and a new `-002/depth` output. The old partial run is preserved; the
scientific source, cases and numerical settings are unchanged. The local
supersession record and exact depth driver are included.

Eight handoff inputs are pinned in `handoff-inputs-0399cb2.json` and checked again
immediately before production submission. The submission intent, scheduler stdout
and stderr, and actual job record are preserved. The final regression evidence
path is tied to the full frozen source commit; prior-source regressions cannot
substitute for it. The launch/output guards prevent blind resubmission.

After both gates pass, production submission is recorded under
`/scratch/urbanke/lsa-alt2027/runs/production-launch-0399cb2-001/` and runs are
written under `.../runs/production-0399cb2-001`. A postprocessing job is queued
with `afterok` on the entire production array. Using the same frozen source and
Linux environment, it merges all 7,754 jobs, regenerates the paper reports/assets,
and verifies nine merged Runs plus the report Run. It retains command logs,
manifest checksums and separate immutable outcomes. Its resources are 8 CPUs,
48 GiB and four hours. No Overleaf edits or result publication occur in this handoff.

`record.json` records exact identities, dependency roles, archive provenance and
remote working paths. `SHA256SUMS` covers every stored file except itself. The
source tree is unchanged by this artifact-only archive. Scratch paths are working
locations; durable publication of completed results remains a later milestone.
