# Sealed-store development evidence, 8 October 2026

This archive preserves development decisions and failures. It is **not numerical
admission of the extended store**, a completed ALT experiment rerun, or a uniform
accuracy guarantee. Original Run records and failed rows are retained unchanged.

The offline store built from clean source
`83224dc756481ab8863855a8def26f1504447b42` completed on SCITAS as job `40187608`
with 48 workers: scheduler elapsed **00:08:56**, exit `0:0`. It contains all
137 levels 2–138, 50,553 columns and **11,633,407,536 payload bytes** (about
12 GB). The sealed plan and manifest bind the builder and corrected offline
contour sources. The store remains outside Git at
`/scratch/urbanke/lsa-alt2027/stores/sealed-83224dc-20261008-001`.

The scheduler elapsed time includes the complete job. The wrapper records
516.983901 seconds for its builder command (including store verification and
sealing), and 518.022991 seconds including the final identity checks. The
scheduler's `40187608.0` MaxRSS is 5,776,564 K. The wrapper's child maximum RSS
has a different scope and is explicitly not simultaneous aggregate worker RAM.
No hardware model is inferred. These records certify completed construction
and integrity sealing; reader validation and native integration are separate.

## Contents and interpretation

- `runs/`: lossless archives of the original failed six-level kernel pilot and
  both immutable profile pilots, including saved samples, evaluation arrays,
  summaries, Run manifests and available source capsules. Every original
  output listed by each Run manifest was hash checked. The first kernel pilot
  preserves all 2,328 rows: 395 failed, 502 precision limited and 1,431 passed
  under its original assessment. Its maximum discrepancy was 13,335.74 nats.
- The profile pilots ran with a dirty development source tree and remain
  diagnostics. Neither originally saved a source-capsule directory. Their
  `recovered-source-capsule/` supplements contain only bytes matching the exact
  source hashes in the original Run. `source-capsule-recovery.json` records the
  recovery origins and unavailable historical files; it does not assert that
  the original execution was clean or that the recovery is complete.
- `investigation/count-coordinate-002/`, `003/`, `004/` preserve the interpolation
  comparisons and their exact scripts. Physical float64 grid coordinates,
  balanced count stencils, centered `log1p` coordinates, and shifted transition
  interpolation were investigated separately. The early precision probe is
  historical evidence from intermediate candidates, not the final reader.
- `investigation/count-hybrid-sweep-001/` and `003/` preserve the denser tests
  that rejected too-early branch handovers. Sweep002 allowed points above the
  sealed upper bound and is excluded from accuracy evidence, as described in
  `count-interpolation-findings.txt`. The empty failed coordinate001 attempt
  yielded no numerical result. Large JSONL rows are losslessly gzip compressed;
  the archive manifest records both stored and uncompressed hashes.
- `investigation/count-remaining-reference-001/`, `count-error-decomposition-001/`
  and `count-payload-decomposition-001/` retain independent 45/60-digit query
  references and 192 independent 60-digit stored-node references. Two nominal
  3e-9-nat misses remain at L2/r1000000/u−0.2573 and L138/r990031/u−4.013. The
  decomposition identifies stored payload rounding as the main contribution;
  high-precision interpolation of those payloads rounds to the same answers.
  Those failed checks have not been removed or retrospectively relabeled.
- `investigation/sealed-native-probe/` contains the pinned Python reader,
  standalone C prototype, small compiled libraries and build identity, exact
  diagnostic/production-API comparison scripts, timings and boundary records.
  The recorded 2,328 scalar and 19,224 batched comparisons plus selected full
  profiles establish finite Python/native parity on those tests. They do not
  certify the underlying store's absolute accuracy. Timing scopes and compiler
  identity are retained in the records and `alt-sealed-native-evidence.txt`.
- `cluster/`: full-store plan, sealed manifest, submission, scheduler completion
  and wrapper launch/exit records, plus the old direct-reference completion
  manifest/result. Completion of the old reference calibration does not admit
  the new sealed reader. Kernel-store payloads are deliberately excluded.
- `working-records/`: corrected diagnostic, pilot logs/configurations and
  completed implementation checks available at archive time. The final local
  native integration regression suite passed **544 tests in178.91 seconds**;
  its full log and the completed full Appendix check log are included. These
  implementation checks do not replace numerical admission of the full store.
  Further cluster validation belongs in separately dated records.

`manifest.json` binds every archived file by byte size and SHA256, every member
of each Run archive, and each decompressed JSONL file. `archive_evidence.py`
records the selection, source-recovery and verification procedure. The manifest
cannot include its own hash. All archive and member hashes were verified after
writing. Local source paths are working locations, not public archive URLs.
