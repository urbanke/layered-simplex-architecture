# Depth calibration records — 7 October 2026

These archives preserve the completed assessment of the original depth run and
independent checks of the adaptive upper-window fix. They are finite numerical
validation records, with original failures retained. They do not authorize
production experiments.

`archive-index.json` identifies the packages. Every package has a per-file
SHA-256 manifest, and `SHA256SUMS` covers the archives and manifests. The packaged
input files and execution-source capsules were verified against their saved
hashes. Packaging-time commit observations are labelled separately from hashes
recorded during execution. Public durable retrieval remains a release task.

| Record | Result and scope |
|---|---|
| Original completed-case assessment | 26 of 27 declared cases read; 24 pass all component/mixture loss and mass gates after 23 supplemental grid groups; two original depth-80 upper-window failures retained; full Bible case pending in this snapshot |
| Repaired cases and grid assessment | Both original failed profiles pass at all eight declared depths; four supplemental groups complete actual grid halving, with component/mixture loss and raw-mass gates passed |
| Adaptive-window profiles | Three saved profiles at depth 80; the two failing profiles agree with fixed upper windows 60/80 and finer quadrature; a previously passing profile is bitwise unchanged |
| Expanded-window kernel checks | All 16 independent high-precision checks pass at depth 80, counts 0/1/9/40, and log-grid coordinates through 80; maximum direct residual 2.88e-13 nats |
| Adaptive-window regression | Full appendix script and 260 tests pass; initial sandbox shared-memory errors and successful permitted rerun are both preserved |

For the 24 passing original cases, the maximum final component codelength
change is **1.784e-10 bits/token**; the maximum mixture change is
**3.799e-11 bits/token**. The maximum component predictive KL change is
**3.931e-12 bits**, the maximum mixture predictive KL change is
**2.042e-12 bits**, and the maximum raw probability-mass error is
**2.726e-9**. These have different units and are not interchangeable.

The original failures are `uniform-n20000` and `dirichlet_half-n20000`:
at depth 80 the initial upper boundary 35 ended before the significant mass.
The original assessment retains these failures rather than changing the saved
results. The adaptive implementation expands the shared family window under
bounded, recorded rules. Its profile checks show a maximum raw-mass error of
1.151e-11 and maximum KL change of 2.078e-11 bits across the saved comparisons.
The separate repaired-case run then evaluated both original failed profiles on
their complete eight-depth grids using the saved original samples. Its final
assessment passed all actual grid halvings, component and mixture loss gates,
and raw-mass checks. Maximum component KL change was 8.512e-12 bits; maximum
mixture KL change was 4.437e-14 bits; maximum raw-mass error was 2.049e-11.

The original `depth.py` execution hash is
`bb4fa35e23cd785c4bef5c44bcf616a8565834ac8a4f494571b9ea84e01dbdd5`.
The tested adaptive-window implementation hash is
`e6ce3760a35febf776d7a9fc1fb494f0ea98375af6d72028a0f630bbe383313e`.
The latter matches the implementation committed in
`06d393de43b7e8348d6594fa3143f737eb7d1537`; packaging records distinguish this
later commit observation from the source hashes measured by the validation runs.

The full Bible profile (`n=915860`, `d=100000`, depths 0–54) is being completed
from its saved sample in a separate immutable run, with default/refined evidence
and actual-grid-halving supplements checkpointed per depth. Its results will
receive a separate completion record. The combined coverage assessment,
post-integration regression checks, sampling/runtime pilot, protocol freeze,
and fresh production campaign remain subsequent acceptance work.
