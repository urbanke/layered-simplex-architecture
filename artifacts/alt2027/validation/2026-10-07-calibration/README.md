# Numerical calibration milestone — 7 October 2026

Five completed finite validation runs are archived here. `archive-index.json`
identifies their configurations, files, source records, and checksums; each archive
has an accompanying per-file manifest. `SHA256SUMS` covers the archive packages.
The source records distinguish hashes measured during execution from the
explicitly labelled packaging-time observations.

| Completed run | Recorded result |
|---|---|
| Independent kernel grid | 263/263 direct and batched cases pass; depths through 138; counts through 1,045,889; low-count maximum residual 1.46e-12 nats; overall maximum 1.09e-9 nats |
| Special functions | 36 Meijer-G comparisons through depth 3 within 1e-44 nats; 12 positive depth-two recursion checks agree at the saved 45-digit precision |
| Independent prior/identity suite | 44 checks pass; six 200,000-draw Monte Carlo cases within 1.731 standard errors; direct simplex integration through depth 3; full small-alphabet identities |
| Power refinement | 11 targets × 81 powers × two numerical settings; 1,782 component evaluations; maximum component KL change 4.69e-12 bits; four independent power-two references pass |
| Bible chain | First 2,000 canonical tokens, depths 0–54; maximum prediction discrepancy 1.38e-13; cumulative codelength discrepancy 9.10e-11 bits |
| Batching pilot | Five saved uniform profiles at five depths; first-profile probability difference 5.21e-15 versus an individual evaluation; original labels and diagnostics retained |

The kernel investigation identified a substantial error in the candidate
high-depth saddle/series shortcut. The ALT adapter now uses direct contour
evaluation and a certified small-t expansion there. These observations concern
the candidate numerical implementation; the archives do not establish which
implementation generated historical paper values.

Both sampled stores pass the original 3e-9-nat row criterion. One depth-53
zero-count row has a 1.42e-11-nat residual, recorded separately against the more
stringent new low-count criterion. Profile-level checks assess its practical
amplification. Stores remain unchanged.

## Remaining before production

The 27-case depth/profile refinement run is still active at this milestone.
Its completed profiles, final Bible/spectrum cases, actual-grid step comparisons,
and both component/mixture loss gates will receive a separate assessment. These
archives therefore provide completed bounded checks, not a production admission
record. The sampling/runtime pilot, combined calibration assessment, protocol
freeze, complete fresh experiment campaign, and manuscript asset integration
remain subsequent milestones.

The final integrated revision `d1e4d26` passed all **246 repository tests**,
including the benchmark batch integration. The full appendix validation script
passed. The exact logs are `tests-d1e4d26.txt` and `appendix-validation.txt`.
These regression checks are separate from the archived numerical runs above.
