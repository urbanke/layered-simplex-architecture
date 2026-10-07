# Distributed smoke and SCITAS preflight — 7 October 2026

This milestone preserves the completed end-to-end smoke campaign and the passed
SCITAS compute-node preflight for scientific source commit
`cc1524a636a5f1996b4fbff1668c477ed0e989b8`. The 62 planned jobs completed with two
local worker processes; all nine merged experiment Runs and the report Run also
completed. Every Run inventory, file size, SHA-256 and saved protocol identity
was verified both in its working location and after extracting the capsule.
These are smoke and infrastructure checks, not production estimates.

The scientific source capsule contains the exact 178 files recorded by those
Runs, including the pinned corpora and environment lock. Its source-tree hash is
`cefc1852613bb1397a881790a7a769e75425e41c437c270eecfa9fbaec7a364d`.
The smoke capsule includes the common samples, raw trials, block diagnostics,
merged summaries, rendered reports, source-job mappings and controller records.
The records retain their original local paths; checksums establish the archived
identities independently of those paths.

The setup capsule preserves the full development regression log (374 tests and
7 subtests passed) and full appendix validation (all eight checks passed), both
from before the final job-ordering fix. The targeted distributed regression
after that fix passed 109 tests and 7 subtests. The earlier broad regression must
not be read as a full-suite run at the final commit. The defect was an
inconsistent job ordering after JSON serialization sorted experiment keys; the
final ordering fix and its regression test are included in the source capsule.

Slurm preflight job `40152542` failed at the exact-partition guard before any
numerical work. Its log is retained. Corrected preflight job `40152660` exited
zero with source and inputs unchanged. Its three JSON records preserve the
compute-node runtime and the complete numerical-store verification: all 106
files, 2,802,407,082 bytes, were checked against their declared SHA-256 values.
This verifies installation, source and store identity, not numerical accuracy
across the production domain. Calibration configurations included in the setup
capsule record preparation only; no submitted calibration outcome is claimed.

`archive-index.json` lists the three compressed capsules. Each has a per-member
size/SHA-256 manifest; `SHA256SUMS` covers all archive files except itself.
`verification.json` records all 72 verified Runs and the extraction check.
`excluded-bulk.json` preserves sizes and hashes for the two Git-transfer bundles
and the exact-reference subset tarball, whose bulk bytes are not duplicated.
The subset's own per-file manifest is retained. The full 2.80 GB store is likewise
identified by its complete preflight hash inventory rather than copied here.
Public durable retrieval remains part of the final release archive.
