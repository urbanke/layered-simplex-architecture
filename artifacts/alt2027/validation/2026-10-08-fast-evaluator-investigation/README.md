# Fast-evaluator investigation, 8 October 2026

The user identified a mismatch with the fast PMWM computation strategy. This
archive records the source audit and bounded native/table diagnostics. Read
`outputs/alt-fast-evaluation-findings.txt` for the findings and measured scope.

The high-depth approximation had genuine accuracy failures. Reusable accurately
computed columns nevertheless recover large speedups on the tested calls:
about 68–71x for two selected five-depth prediction/evidence evaluations after
preparation. These are diagnostic measurements, not production acceptance or
full-campaign speed estimates. Preparation is recorded separately.

The production handoff 40162230 was placed on a reversible user hold; numerical
validation 40160952 was left running. No model, sampling, frozen numerical code,
original store, or Overleaf text was changed. The frozen execution source remains
0399cb28b973e5550fcbcbce082e8d36c95c2e1e. A new evaluator needs its own frozen
source/configuration, completed validation, timings and production plan.

`manifest.json` hashes the archived notes, scripts, summaries and two sampled
profile inputs/results. Native shared libraries and bulk diagnostic columns stay
outside Git; their source/data hashes and local paths are in the saved records.
The first profile driver stopped because it used the wrong result attribute;
its exception log is preserved and the corrected completed run is separately
named 002. The numerical source was unchanged by that driver repair.

The diagnostic scripts retain their original workspace paths and require the
pinned source/library environment; they are experimental probes rather than a
standalone released reproduction entry point.
