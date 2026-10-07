# Numerical engine for the ALT experiments

The ALT evaluator lives in `src/lsa/alt/depth.py`. It uses a pinned scientific
subset of `product_model_with_memory`, with the existing independent LSA
implementation retained for regression and cross-checks. The power model has a
separate evaluator in `src/lsa/alt/powers.py`.

## Source and model identity

The vendored upstream revision is `240406d16be0e7c5dcd7d2ee0e14d5ee4f28c915`.
`src/lsa/alt/_vendor/pmwm/provenance.json` records upstream and current file
hashes. Local adaptations include explicit immutable-store operation, preserved
integration diagnostics, and the corrected high-depth contour calculation.

The ALT adapter implements the exact requested depth grid, including the uniform
atom at depth zero, and equal prior weights over that grid. Sequence evidence
sets the posterior weights. Predictions use base/augmented count-profile
families. Both component and mixture probability sums are checked before the
explicit final normalization used by the benchmark protocol.

A single powered layer is `Y=E**w`; its power grid is separate from the unit-power
depth grid. Powers zero and one have analytic endpoints. The remaining powers
use positive quadrature with recorded tail and convergence diagnostics.

Each configuration identifies the implementation, numerical settings, store
contents, runtime/library versions, and actual execution path. The current ALT
path uses Python/scipy scans without native-kernel evaluation or depth-tail
truncation. Thus native/sparse/truncated-path comparisons are outside the path
used for these experiments.

## Immutable kernel store and direct contour route

`experiments/alt2027/store-candidate.json` identifies all 106 files of the
2.80 GB `anchors_prod` store. The local engine configuration supplies its path;
configuration identity uses the file hashes and is independent of that path.
Every used file is checked, and the adapter cannot build or modify the store.
The separate `probe_exact` store supports independent sampled-row comparisons.

Stored rows cover depths 2–53. The original high-depth saddle/series shortcut
failed reference checks. It was replaced with vectorized direct contour
integration and a strictly bounded small-t expansion. The legacy configuration
key `saddle_min_depth=54` now selects this direct-contour route; saved diagnostics
name the route `direct-contour`. Both the original shortcut residuals and the
corrected results remain in the validation records.

The outer log-grid starts at the configured spacing. If any member of a related
profile family has an unresolved narrow peak, the evaluator retries the entire
depth at half the spacing, down to `minimum_grid_step`. Parent and augmented
profiles always share the same grid. A remaining unresolved peak, tail failure,
or invalid probability mass stops that evaluation. Recorded diagnostics retain
the requested/actual spacing, number of refinements, and integration boundaries.

## Declared calibration suites

The five `*-validation.json` specifications in `experiments/alt2027/` freeze
cases, seeds, precision settings, refinement rules, and tolerances before runs.

| Suite | Evidence produced |
|---|---|
| Kernel | 45/60-digit independent contour references through depth 138; count/interpolation boundaries; pointwise/batched columns; Meijer-G and positive depth-two recursion; sampled stored rows |
| Prior | Independent prior simulation with saved seeds and standard errors; direct simplex integration through depth 3; complete small-alphabet sequence, prediction, KL-chain and discovered-set identities |
| Depth | Saved paper-domain profiles, all 81 benchmark depths on each target, large-alphabet and Bible cases; outer-grid/window refinement; component and mixture loss changes; raw normalization errors |
| Power | All 81 powers on a saved profile from each paper target; stricter kernel/outer quadrature; independent power-two references; full component and posterior records |
| Chain | First 2,000 canonical Bible tokens, all 55 depths; family prediction versus separately scanned profile evidence, with every transition retained |

These are finite, explicitly recorded numerical checks. Numerical error is
reported in its actual units: log-kernel nats, total evidence bits, bits/token,
next-symbol KL bits, or probability mass. The profile-level performance target
is stability within `1e-5` bits; a raw-mass guard is reported separately. The
original stored-row criterion is `3e-9` nats. A stricter `1e-11` low-count direct
kernel check additionally records small stored-row residuals; those residuals
are assessed with the profile-level refinement results.

## Running and verifying checks

From a fixed checkout with the locked environment:

```sh
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python scripts/alt_experiments.py configure-engine --store-directory /path/to/anchors_prod --out output/alt2027/engine-candidate.json
python scripts/alt_experiments.py calibrate --suite prior --config experiments/alt2027/prior-validation.json --out output/alt2027/prior-001
python scripts/alt_experiments.py calibrate --suite kernel --config experiments/alt2027/kernel-validation.json --out output/alt2027/kernel-001
python scripts/alt_experiments.py calibrate --suite depth --config experiments/alt2027/depth-validation.json --engine-config output/alt2027/engine-candidate.json --workers 4 --out output/alt2027/depth-001
python scripts/alt_experiments.py calibrate --suite power --config experiments/alt2027/power-validation.json --workers 2 --out output/alt2027/power-001
python scripts/alt_experiments.py calibrate --suite chain --config experiments/alt2027/bible-chain-validation.json --engine-config output/alt2027/engine-candidate.json --out output/alt2027/chain-001
python scripts/alt_experiments.py verify output/alt2027/prior-001
```

Use explicit local store paths in the engine and kernel configurations. Keep
file identities fixed when moving stores. Each output directory is immutable.
The command wrapper records the complete source tree, installed environment,
inputs and results, rejects changes during execution, and returns failure if a
declared check fails. Worker sharding preserves each case's seed coordinates.

The depth suite also checks the actual grid spacing saved by each integral.
Where adaptive retries made both nominal settings equally fine, the assessor
evaluates another half-step using the same saved profile. It gates both the
component and mixture loss changes, plus raw probability mass. Existing runs
can be assessed with `assess-depth --sources DIR --config FILE --engine-config FILE
--supplement --out NEW_DIR`. An incomplete snapshot lists its pending cases;
original failures remain visible.

A passed suite supports its saved cases. Production admission additionally
requires the combined coverage assessment, a frozen protocol, a clean source
commit, matching engine identity, and complete required checks. Calibration
summaries alone cannot enable the production runner.

## Batching and regression

`batch_depth.py` shares kernel preparation across bounded cohorts and caches
compact exchangeable count-class results. It maps predictions back to the
original label order and preserves diagnostics. Evidence-only experiments stream
bounded chunks of saved profiles. Cache keys include alphabet size, the complete
count profile, and the ordered depth grid.

Batch/individual comparisons cover analytic endpoints, independent depth-two
checks, a small pinned store, and saved paper-domain pilot profiles. Numerical
changes also require the full `scripts/validate_appendix_c.py` and test suite.
The exact dependency lock is `requirements-alt.lock`; every new run records its
installed versions, hardware, worker count, and nested-thread settings.
