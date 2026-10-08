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
checks high-count kernels through `r=967,w=80` against an 80-digit direct
exponential-coordinate integral with explicit concave-tail bounds, and checks
that deliberately insufficient quadrature raises an error. The direct
simplex reference uses neither the Laplace-kernel formula nor the implementation's
outer quadrature. An additional high-dimensional `w=2` reference evaluates the
closed-form Gaussian/exponential integral and its moment recurrence at 80 and
100 decimal digits, followed by separately refined outer quadrature. Its
restricted count/window domain is explicit; it supplies an independent kernel
calculation rather than reusing the production kernel quadrature.

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

## Saved-profile calibration pilot

`lsa.alt.power_validation.run_power_validation(config, output_dir)` evaluates
one saved profile from each configured target at every declared power under
default and stricter settings. The paper-domain specification is
`experiments/alt2027/power-validation.json`: `d=10000`, `N=1000`, all eleven
targets, powers `0,...,80`, and power-table trial-zero seed coordinates. Thus the
pilot includes both diffuse targets and concentrated Zipf profiles, with the
same Dirichlet target draws as the power benchmark's first trial.

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m lsa.alt.power_validation \
  --config experiments/alt2027/power-validation.json \
  --out output/alt2027/power-calibration-001
```

All target/count arrays are saved before numerical evaluation. The pilot saves
each component's raw classwise probabilities, sequence log-evidence, convergence
diagnostics and elapsed time immediately. It compares absolute sequence
log-evidence changes in bits and absolute target-weighted predictive KL changes
in bits to `1e-5`; the latter is the difference of cross-entropies, so the target
entropy cancels. Raw predictions are used throughout, without normalization.
Posterior total variation, effective component counts, upper-grid posterior mass
and mixture stability are retained. A failed component leaves an explicit record
and prevents that profile's mixture from being accepted; remaining calibration
components can still be evaluated to diagnose the failure's extent.

The independent closed-form `w=2` check covers the saved uniform, step and both
Dirichlet-target profiles, whose maximum counts are inside its bounded recurrence
domain. The driver hashes its numerical source dependencies at the start and
finish; changes invalidate the run. Per-profile records and a final file manifest
are immutable. A passed pilot remains a finite-profile validation record, not a
certificate for every possible sample or a production result.

The 7 October 2026 pilot at `output/alt2027/power-calibration-001` passed all
1,782 component evaluations (eleven profiles, 81 powers, two settings). Its
largest sequence-evidence change was `8.67e-10` bits, largest component
predictive KL change `4.69e-12` bits, largest mixture predictive KL change
`1.94e-13` bits, and largest raw normalization discrepancy `3.77e-12`.
The four independent `w=2` references agreed within `2.97e-10` bits in sequence
evidence. The maximum posterior total variation under refinement was `4.89e-13`.
All scientific dependency hashes and saved sample hashes remained unchanged.

The concentrated targets exercise broad posteriors: Zipf exponents 3, 4 and 5
had effective component counts 32.62, 59.49 and 59.59, respectively. Their
posterior mass on powers 41--80 was 9.15%, 72.43% and 46.54%. No component was
removed or probability vector normalized for these checks.

The two-worker pilot took 1,614 seconds (26.9 minutes) on the recorded machine.
The default evaluations alone totaled 1,471 component-seconds across the eleven
profiles. These are measured pilot timings; fresh profiles and concurrent load
can change runtime. The run's `summary.json` and `files.json` bind its domain,
settings, source hashes and per-profile records. Raw arrays remain local working
artifacts until the campaign's durable archive step.
