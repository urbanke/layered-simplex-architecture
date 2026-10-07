# ALT experiment campaign

The scope is the current `alt.tex`: four figures, five numbered table objects,
the appendix validation table, and the existing in-text controls. Overleaf stays
the authoring source; this repository supplies reproducible experimental assets.

`inventory.json` preserves the original manuscript-to-code audit. The executable
`protocol.json` now resolves its operational settings. `smoke.json` uses small
alphabets and short samples to exercise the entire pipeline. Results from that
protocol are development checks, with their purpose recorded in every manifest.

## Sequence and completion criteria

1. **Implementation and integration (started; first end-to-end run complete).**
   All retained experiment families have runners. Every comparator, the depth-zero
   atom, the depth mixture, and the correct single-layer power mixture are present.
   Reports read saved samples/results and recreate the existing plot/table roles.
2. **Numerical calibration.** Complete the appendix checks and test the actual
   experiment domain. Check grid/contour refinement, tails and the corrected direct-contour route,
   especially depths 54–138 and long Bible profiles. Compare powered kernels to
   independent calculations, then require KL stability within 1e-5 bits under
   numerical refinement. Store raw normalization errors separately. The 1e-3
   normalization rejection threshold is an engineering guard, not an error bound.
3. **Timing and uncertainty pilot; protocol freeze.** The completed 20-trial pilot
   measured runtime, paired uncertainty and rare-trial variation. Protocol v2
   fixes 1000 trials per primary/power benchmark cell and 1000 profiles per
   spectrum/depth cell; factorial retains 5000 profiles per cell. Every comparison
   uses common samples within a trial. Ten disjoint trial blocks provide an
   additional stability diagnostic, with fixed counts and no stopping based on
   favorable results. Models, sample-size grids and plot roles are unchanged.
4. **Fresh production campaign.** Freeze a clean source commit, protocol, locked
   environment, corpus, complete numerical-store hashes, and calibration record.
   Run the retained grids, retaining per-trial records and failed-case diagnostics.
5. **Final assets and manuscript revision.** Regenerate all tables, plots and
   numerical prose from the verified records. Check layout and symmetric rounding,
   upload selected assets to Overleaf, then take a source snapshot and archive
   bulk raw results/stores with durable retrieval locations and checksums.

These are sequential acceptance milestones. The six-day target leaves the final
stage for interpretation and manuscript integration; numerical or sampling
changes are versioned explicitly so they cannot silently alter completed runs.

## Fixed retained workload

| Family | Current production settings | Saved output / paper role |
|---|---|---|
| Architecture | d=24, L=4, seed 7 | all layers and product; fig:lsa TikZ |
| Depth spectrum | 3 panels, 11 alpha values, 1000 profiles; depths through 69/92/138 | 33000 profiles and all evidence; fig:spectrum |
| Main benchmark | 11 targets, 8 sample sizes, 1000 trials; d=10000; L=0..80 | 88000 common samples; tab:bench, fig:orlitsky, tab:posteriors, uniform ablation |
| Power benchmark | independent 1000-trial set; n=1000; powers 0..80 | 11000 samples; tab:powers-bench |
| Bible | original prefixes plus n=100/500; d=100000; L=0..54 | token IDs, dictionary, per-token baseline losses, profile evidence; tab:bible and controls |
| Second tokenization | its original corpus and corresponding prefixes | classical and Dirichlet controls |
| Factorial scaling | 5 d values x 5 N values x 4 alphas, 5000 profiles/cell | 500000 profiles; joint tab:alphabet/tab:data |
| Depth coefficient | 17 c values, alphas 2/3/4, 1000 shared profiles | 3000 profiles evaluated at all depths; fig:depth |
| Validation | declared identities and independent comparisons | measured residuals, units and tolerances; appendix validation |

The spectrum uses the same displayed depth selections as the current asset.
The depth plot uses the three visible alpha curves; the text's additional alpha
1.5 mention is recorded for reconciliation. The historical plotted heuristic
uses B(alpha)=alpha*log2(e)-2. Alphabet slopes are free OLS fits, with the separate
slope-one offset also retained. These details are explicit in the protocol.

Dirichlet targets are redrawn per trial and shared across that trial's sample
sizes. Counts are independent across sample sizes and paired across methods.
The primary and powered tables use independent, explicitly seeded sample sets.
Genuine infinite KL losses and undefined absolute-discounting cases stay distinct
from finite results. All methods use the same reporting and rounding rules.

## Commands

For the laptop and Jed, [the distributed launch guide](../../cluster/alt2027/README.md)
describes pinned environments, compute-node preflight and deterministic jobs.
`scripts/alt_distributed.py prepare` partitions targets/cells and global trial
IDs; `work` writes immutable attempts and verified completed jobs; `merge`
checks complete coverage and reconstructs the normal report input layout.
Workers may restart with a different allocation without changing any draws.
Each campaign root binds one runtime; Linux and macOS outputs remain separate.
The default production partition has 1584 jobs (100-trial blocks, 500 for
factorial), with 10 laptop workers or up to 72 workers per allocated Jed node.
Production still requires the combined calibration certificate and frozen source.

From the repository root, after the setup in the main README:

```sh
python scripts/alt_experiments.py plan --protocol experiments/alt2027/protocol.json
python scripts/alt_experiments.py campaign --protocol experiments/alt2027/smoke.json --purpose smoke --out output/alt2027/smoke-001
python scripts/alt_experiments.py report --protocol experiments/alt2027/smoke.json --purpose smoke --sources output/alt2027/smoke-001 --out output/alt2027/report-001
python scripts/alt_experiments.py verify output/alt2027/report-001
```

A single experiment can be run with `run --experiment NAME` and the same protocol,
purpose and output arguments. `--batch-size` bounds kernel-sharing cohorts
(default 20); sample seeds, model grids, and per-trial reporting stay fixed.
Every output directory must be new. A failed run
remains available for diagnosis. Reruns get a new directory; completed records
are never overwritten. Set `PYTHONPATH=src` if using an uninstalled checkout.

The self-contained reference depth backend supports depths 0–2 on small domains.
The fast backend uses `--engine-config FILE`, whose `store` section matches
`lsa.alt.depth.StoreConfig`. Supply an explicit local store path and all content
hashes from `store-candidate.json`. The recorded candidate store has 106 files,
2.80 GB, and numerical columns through depth 53; depths 54–138 select the declared
direct-contour route. The legacy saddle shortcut was replaced after independent
reference checks. Every read is checked against the
pinned store identity, and the adapter cannot build or modify the store.

Production additionally requires protocol status `frozen`, a clean source tree,
and `--calibration FILE`. A passed calibration must bind the protocol, source-tree
hash, complete per-experiment engine hash (including powered settings), runtime,
and covered experiment IDs, with `required_checks_complete=true`. The current
validation driver deliberately produces partial status until the remaining
checks are completed; it cannot certify itself from a smoke run.

## Declared validation work

The base driver measures endpoint identities, exchangeability, raw predictive
normalization, independent depth-two simplex/kernel calculations, discovered-set
KL decomposition, and the naming-only lower-bound counterexample. The dedicated calibration suites now implement:

- Mellin rows through depth 138 on a recorded r/t grid: step halving and an
  independent Meijer-G comparison.
- The 45-digit recursion/contour comparison at depth 2 and r<=3.
- Prior Monte Carlo and simplex integration for depths through 3, with saved
  seeds, counts and cases; report measured errors/standard errors from the rerun.
- The complete small-alphabet independent-reference identity suite.
- The implemented Bible chain check over the declared 2000-token prefix using
  the production numerical settings.
- Full-domain refinement, tail and interpolation checks, including powered
  predictions and sparse/concentrated benchmark profiles.

Run each dedicated suite with `calibrate --suite NAME --config FILE --out DIR`;
see [the numerical guide](../../docs/numerics.md) for exact commands, store
settings, and coverage. A suite records its finite cases and measured residuals.
The production gate additionally requires the combined assessment and frozen
protocol.

The previous manuscript's numerical agreement values are comparison points.
The final validation table will be generated from the new measured results.
No manuscript claims are changed while this implementation work is underway.

## Records and storage

Each run records the protocol and input hashes, exact command, source commit and
file hashes, installed package versions, machine/thread settings, timestamps,
engine settings and store identity. It rejects source/input changes during a run.
Reports verify all source records before reading them; production reports require
one scientific source tree. See `artifacts/alt2027/README.md` for publication
archiving. Local ignored outputs are working storage; public retrievable archives
remain part of the final release milestone.
