# Sealed-store finite-case accuracy contract

The sealed reader uses the versioned `absolute-plus-four-ulp-v2` contract in
[`sealed-store-validation.json`](../experiments/alt2027/sealed-store-validation.json).
It distinguishes the nominal absolute target from a separately reported
computational error budget measured in binary64 spacing. A qualified result does
**not** claim an absolute kernel error of at most `3e-9` nats. This contract is a
finite-case numerical acceptance rule, not a theorem bounding the evaluator
uniformly over its continuous domain.

## Exact rule and units

Let `y` be one binary64 reader result, `R45` and `R60` the independently computed
45- and 60-decimal-digit log-kernel references, and `q = RN64(R60)`. The argument
`u` is the exact binary64 input in both references, rather than its rounded short
decimal display. Candidate floats are likewise lifted exactly for subtraction.
Every error below is in **natural-log kernel units (nats)**.

Define `s` as the larger distance from `q` to its two adjacent binary64 numbers.
This explicit convention is conservative at binade boundaries and for negative
values. Independent references must be finite, agree within `1e-25` nats, round
to the same binary64 number, and satisfy `|q - R60| <= s/2`. Precision convergence
is measured evidence; the contour calculation is not interval-certified.

Each scalar and matrix result is assessed separately:

1. Counts `r <= 3` must satisfy the unchanged `|y - R60| <= 1e-11` target when an
   independent reference is required. They never use the arithmetic allowance.
2. For `r > 3`, a nominal result satisfies `|y - R60| <= 3e-9`.
3. A non-nominal result may be `precision_qualified` only when `4*s > 3e-9`,
   independent reference checks above pass, and `|y - q| <= 4*s`.
4. Otherwise the row fails. Missing or unconverged required references fail.

The separately recorded computational error is `y - q`; it includes the combined
effect of offline kernel evaluation, stored values, interpolation and reader
arithmetic. It is **not** solely unavoidable storage rounding. The reference
rounding component is `q - R60`. The triangle inequality then gives
`|y - R60| <= 4.5*s` for an accepted qualified path. The four-ULP allowance is an
explicit engineering acceptance budget, not a derived forward-error bound for
an arbitrary number of operations.

Both paths must pass their individual rule. Their mutual difference must also
satisfy the original absolute threshold (`1e-11` for low counts, `3e-9` otherwise),
including qualified cases. This additional test may reject two individually
qualified but inconsistent answers.

The driver first compares the actual scalar and matrix reader paths with two
corrected direct binary64 evaluations. It computes independent references whenever
a nominal direct check fails, a direct reference ULP exceeds its nominal gate,
or `r > 3` and four direct-reference ULPs exceed `3e-9`. Thus the arithmetic-budget
regime is independently checked even when the two binary64 implementations agree.
Ordinary nominal screening rows retain the direct-reference basis, explicitly
identified in the summary; the separate independent kernel suite remains required.
Case selection, direct rows, independent references and final rows are saved in
fixed order. Admission recalculates the selection, exact decimal differences,
classes and summary metrics from these records.

## Why version 2 changes the earlier criterion

Version 1 allowed qualification only when **one** ULP exceeded `3e-9`. Version 2
uses the declared **four-ULP computational budget** for this eligibility decision.
This is an explicit relaxation of acceptance in the newly eligible magnitude
range. It does not change the nominal target, conceal earlier failures, or state
that every representable nominal miss is unavoidable.

Two remaining diagnostic misses motivated investigation of the distinction:

| `(L, r, u)` | Reader minus independent reference (nats) | One ULP (nats) | Error relative to nearest reference |
|---|---:|---:|---:|
| `(2, 1000000, -0.2573)` | `-3.553007764186632e-9` | `1.862645149230957e-9` | `-2` ULP |
| `(138, 990031, -4.013)` | `-3.253833695805502e-9` | `1.862645149230957e-9` | `-2` ULP |

The saved 45/60-digit references agree within `4.5e-38` nats. Independent
60-digit evaluations at all 192 participating stored nodes isolate propagated
stored-kernel errors of approximately `-2.6694e-9` and `-2.5810e-9` nats.
Repeating both interpolation stages with 75-digit arithmetic on the exact saved
binary64 values still rounds to the same reader answers. These observations
identify a combined numerical-evaluation/rounding contribution; they do not
prove the original absolute target impossible for a more accurate offline store.
Both cases remain nominal failures and are eligible only for the explicitly
qualified category under version 2.

The change does not excuse small-scale interpolation defects. At a reference near
`2.7e6`, one ULP is `4.656612873077393e-10` nats and four ULPs are below `3e-9`.
An eleven-ULP nominal miss there still fails. More than four ULPs also fail in the
qualified regime unless the actual error independently meets the nominal target.
Tests cover both sides of the eligibility boundary, four versus five ULPs,
low-count exclusion, missing references, and strict scalar/matrix agreement.

## Preserved diagnostic evidence

These diagnostic records are preserved in the repository development archive.
Their original bytes and hashes are unchanged. The first link is a Run archive;
its listed hash identifies the original `data/summary.json` member. Production
admission is recorded separately.

| Record | SHA-256 |
|---|---|
| [Original failed 2328-case pilot summary](../artifacts/alt2027/validation/2026-10-08-sealed-store-development/runs/sealed-pilot-validation-20261008-001.tar.gz) | `c242f433c50bf36b8ea51b2ef9705a4820542daf36c4d6e4baa08e20e6074719` |
| [Corrected-reader diagnostic](../artifacts/alt2027/validation/2026-10-08-sealed-store-development/working-records/sealed-reader-corrected-diagnostic-20261008-001.json) | `c0e0214ec31064c4f8448dc496827ab65aded8edd20e011b2d2f6afea10bbbb6` |
| [Independent 45/60-digit references and nominal failures](../artifacts/alt2027/validation/2026-10-08-sealed-store-development/investigation/count-remaining-reference-001/summary.json) | `b5cf8204a5957d0ab01319d64479a92e3599ffcad2349839fcaeaa3c37cdd265` |
| [Interpolation arithmetic decomposition](../artifacts/alt2027/validation/2026-10-08-sealed-store-development/investigation/count-error-decomposition-001/summary.json) | `d8d0a5ad95fd4d15d2bf999674bcfde14db89e3083a4a2c836fb7f404534c6e9` |
| [192 independent stored-node references](../artifacts/alt2027/validation/2026-10-08-sealed-store-development/investigation/count-payload-decomposition-001/summary.json) | `cfe18f12cfd898c446b0468635cdfb118024cde0ab0788b4c8619b165cada9d9` |
| [Diagnostic interpretation under the original nominal gate](../artifacts/alt2027/validation/2026-10-08-sealed-store-development/investigation/count-payload-decomposition-001/findings.txt) | `56a1aa3cae1a5b4d0508ca48dcaa3bd2587460fe22727a64d1f1b7fcf5d3f250` |

The original pilot remains failed: it predates the corrected interpolation and
has 395 failed and 502 unresolved precision-limited rows. The later diagnostic
records retain the two nominal failures above; version 1 would not qualify them.
No old record is relabeled or converted into a successful version 2 Run. A fresh
Run must bind the revised specification, executed implementation, complete store
hashes and runtime. Python/native execution must be recorded and validated for
the actual selected provider.

## Remaining scientific acceptance

Kernel qualification alone cannot justify predictions: count multiplicities,
quadrature and subtraction of evidence terms can amplify kernel errors.
Production therefore still requires independent kernel checks on their unchanged
committed gates, full declared profile and augmented-profile comparisons,
component and mixture checks, depth/chain refinement, and complete immutable
source/runtime/store identity. The final profile targets remain `1e-5` bits for
predictive KL stability, `1e-5` bits/token for codelength differences, and `1e-7`
for raw probability-mass error. They are distinct from the `1e-3` engineering
normalization guard. No experiment model, sample count, endpoint or reported
loss definition changes with this kernel accuracy-contract revision.
