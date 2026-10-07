# Single-layer powers: reference evaluator

`src/lsa/alt/powers.py` evaluates the family in Appendix G of the ALT paper:
draw independent `E_i ~ Exp(1)`, set `Y_i = E_i**w`, and normalize across the
alphabet. It does not replace a power `w` with `w` independent layers.

```python
from dataclasses import asdict
from lsa.alt.powers import PowerEvaluator, PowerSettings

evaluator = PowerEvaluator(PowerSettings())
result = evaluator.predict(counts, powers=range(81))
settings_record = asdict(evaluator.settings)
```

The explicit grid receives a uniform prior. Results retain component sequence
log-evidences in **nats**, raw component/mixture probabilities, posterior weights,
and numerical diagnostics. No component is skipped and no predictive
normalization is applied inside the evaluator. The benchmark records any
subsequent declared normalization separately. `prediction_by_count(d, partition,
powers=...)` avoids materializing the full alphabet; its probability columns
follow `count_values`, including zero only when the profile has unseen symbols.

## Integral and checks

For `w=0`, evidence is `d**(-N)` and prediction is uniform. For `w=1`, the
Dirichlet(1) evidence and Laplace predictor are analytic. Empty samples and the
one-symbol alphabet are also analytic for every power.

For other integer powers define `v = -log(t)/w` and
`psi_r(v) = t**r E[Y**r exp(-t Y)]`. Integrating over `s=log(E)` gives the
positive log-integrand

```
s + w*r*(s-v) - exp(s) - exp(w*(s-v)).
```

Its unique mode is found by a bracketed solve. Gauss–Legendre quadrature is
refined around the mode and, for the zero-count row, around the potentially
sharp cutoff `s≈v`. Tangents to the concave inner log-integrand give a tail
estimate. Scaling by `t**r` cancels the otherwise large `N log(t)` term in the
outer profile integral. Evidence and all next-symbol ratios share that outer
integral. Adaptive vector quadrature is repeated with tighter tolerances and
an expanded window. Each component's uncorrected predictive mass must pass
the normalization tolerance.

`PowerIntegrationError` stops an evaluation when kernel refinement, a kernel
tail estimate, outer quadrature/refinement, or predictive normalization fails.
Settings, windows, quadrature evaluations, observed kernel refinement errors,
their alphabet-amplified estimate, and raw probability sums are recorded.
These are numerical convergence diagnostics, not rigorous interval-arithmetic
error certificates. Normalization alone does not establish accurate evidence;
independent reference checks are therefore part of validation.

## Bounded validation and limits

The default intended domain is the retained powered benchmark: `d<=10000`,
`N<=1000`, and integer powers `0,...,80`. Larger alphabet/sample limits require
explicit settings and additional validation. The evaluator is a reference path;
its timing is not yet a production performance guarantee.

`tests/test_alt_powers.py` checks analytic endpoints; the independent `erfcx`
formula for the `w=2` zero-count kernel; direct integration over the original
two- and three-symbol simplex; evidence-ratio and mixture identities; and
`w=80` at `d=10000,N=1000` under stricter kernel/window/outer settings. It also
checks that deliberately insufficient quadrature raises an error. The direct
simplex reference uses neither the Laplace-kernel formula nor the implementation's
outer quadrature. The high-dimensional test checks convergence and normalization,
not agreement with an independently computed high-dimensional evidence.

The focused tests are run with:

```sh
PYTHONPATH=src python -m pytest -q tests/test_alt_powers.py
```

An additional bounded check on 7 October 2026 evaluated all 81 powers on the
eight-symbol profile `(4,2,1,1,0,0,0,0)`. Every log-evidence was finite, the
largest raw normalization discrepancy was `2.3e-14`, and the run took about
57 seconds on the development machine. At `d=10000,N=1000`, powers 2 and 80
were also checked on 1,000 singletons and the `(999,1)` profile; their largest
raw normalization discrepancy was `7.1e-13`. These selected evaluations took
roughly 0.4–0.6 seconds per component. A five-class concentrated profile took
roughly 1.1–1.4 seconds per component. They are calibration/timing observations,
not paper experiment results or runtime predictions for all count profiles.

Before accepting production results, run the repository's full required numerical
checks and retain the numerical diagnostics from every evaluated trial/component.
Independent high-dimensional checks and timing on the complete frozen benchmark
profile distribution remain part of that acceptance gate. Do not infer complete
domain calibration from the finite validation profiles.
