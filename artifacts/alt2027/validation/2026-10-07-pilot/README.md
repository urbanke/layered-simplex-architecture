# Timing and paired-uncertainty pilot — 7 October 2026

This archive contains the completed validation-purpose run from clean, detached
commit `d1e4d2693610595d43434a6079e2162176e3c149`. Its Run manifest and every saved
output were verified. The pilot retained all nine primary methods, depths 0–80,
fixed depth 22, the original primary seed and common-sample rules. It used
`d=10000`, `n=1000`, and 20 trials each for uniform and Zipf-5: 40 trials in total.
This two-target subset does not change the production protocol.

`timing-uncertainty-d1e4d26-001.tar.gz` includes the complete run, labelled samples,
per-trial diagnostics, derived protocol, engine configuration, analysis script,
all method and paired summaries, and a verified scientific source capsule.
Its accompanying manifest lists every member's size and SHA-256. `SHA256SUMS`
also covers the packaging files. The numerical store is identified by its full
file hashes; its 2.80 GB contents are not duplicated here.

The recorded run took 138.76 seconds, including 11.22 seconds of shared depth
preparation for uniform and 126.82 seconds for Zipf-5. Batch size was 20 and
numerical thread limits were one. The Run timer starts after store opening;
concurrent regression checks could affect wall time. These two samples of the
domain do not establish runtime for every target, sample size, or experiment.
The maximum raw component probability-mass discrepancy was 2.57e-10.

All nine methods use the same mean/SE rule; the analysis contains all 18 method
cells and all 72 unordered paired comparisons. SE is sample standard deviation
divided by sqrt(20), with paired SE computed from within-trial differences.
For Zipf-5, representative comparisons in both directions are:

| Difference (left minus right), bits | Mean | Paired SE |
|---|---:|---:|
| Dirichlet mixture minus depth mixture | -0.00016423 | 0.00025320 |
| Good–Turing hybrid minus depth mixture | +0.00047945 | 0.00026780 |

Neither point difference alone establishes a general ordering. Absolute
discounting on Zipf-5 produced 2 finite, 14 infinite, and 4 undefined losses;
these trials remain in the records, and no finite-only mean is substituted.

On uniform data, depth-mixture loss was 0.000034799 ± 0.000034799 SE bits.
One trial contributed 0.000695979 bits; the other 19 were at most 1.56e-9 bits.
This illustrates the sampling variability of the current 20-trial estimate.

At this milestone, both production benchmark trial counts remain 20. An increase
to 200 has been proposed for author approval; it has not been applied by this
archive. No manuscript values or production results are changed here.
