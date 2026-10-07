# A Layered Simplex Architecture for Large Alphabets

Research code and artifacts for the ALT revision of the paper by Meir Feder,
Yaniv Fogel, and Ruediger Urbanke.

The current manuscript is **`alt.tex`**. Authors edit it in Overleaf; this
repository stores manual source snapshots, numerical implementations, experiment
specifications, validation records, and the results used to produce the paper.

## Start here

- [Manuscript snapshots](manuscript/README.md): the ALT starting point and the
  preserved arXiv source, with compilation instructions and file checksums.
- [ALT experiment plan](experiments/alt2027/README.md): the experiments retained
  in the shorter paper, their current settings, implementation gaps, and the
  decisions to settle before production runs.
- [Numerical engine integration](docs/numerics.md): reuse of the faster evaluator
  from `product_model_with_memory` and the checks needed for the ALT models.
- [Artifact records](artifacts/alt2027/README.md): how a paper result is linked to
  its samples, settings, code revision, numerical store, and generated output.
- [Corpus record](data/README.md): the pinned King James Bible token stream.

## Current stage

The ALT source is prepared and compiles. The first executable campaign is in
`src/lsa/alt/`, with an explicit protocol, all manuscript comparators, the
single-layer power mixture, saved common samples, and report generators for the
retained plots and tables. A small end-to-end campaign exercises every family.

The next milestone is full-domain numerical calibration, followed by the
production protocol freeze and fresh runs. The current protocol has status
`implementation`; the runner enforces this distinction. Detailed status and
commands are in [the campaign guide](experiments/alt2027/README.md).

## Code setup

Python 3.11 or newer for the package; the ALT lock was tested on Python 3.14.6:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-alt.lock
python -m pip install --no-deps -e '.[dev]'
```

The existing numerical checks are:

```sh
python scripts/validate_appendix_c.py --quick
python -m pytest -q
```

The validation script retains its historical filename. Full numerical validation
is required when numerical code changes. ALT validation also covers the additional
models and the full domain of its agreed experiments. The exact dependency lock and installed package versions are recorded with
each run. See `requirements-alt.lock`.

## Repository layout

| Location | Purpose |
|---|---|
| `src/lsa/` | Prior construction, evidence, prediction, baseline estimators, and independent reference calculations |
| `scripts/` | Existing experiment, reporting, and validation implementations; ALT candidate drivers are mapped in the inventory |
| `tests/` | Numerical and implementation regression checks |
| `experiments/alt2027/` | ALT scope, protocol decisions, and experiment inventory |
| `artifacts/alt2027/` | Tracked run records, validation summaries, and selected generated paper artifacts |
| `manuscript/` | Manual Overleaf source snapshots |
| `data/` | Pinned compressed corpus and its metadata |
| `output/alt2027/` | Local per-run samples, losses, logs, and intermediate results; ignored by Git |
| `tables/` | Local numerical caches; ignored by Git |

Both numerical implementations in `src/lsa/` are retained for independent checks.
The state-family, BPE, and online-context drivers and their tests are historical
research tools; the ALT inventory defines the current campaign scope.

## Historical arXiv reproduction

The earlier code, full reproduction launcher, result manifest, and transcribed
paper values are preserved at
[`arxiv-code-2026-08`](https://github.com/urbanke/layered-simplex-architecture/tree/arxiv-code-2026-08)
(commit `e9543dce2dd80078f951072ab182230fae541098`). Its figure/table numbering
and results refer to the longer arXiv version. They remain available for historical
comparison while ALT run records are produced from new executions.

## License and citation

Code is under the [MIT license](LICENSE). [CITATION.cff](CITATION.cff) contains
the authors and arXiv citation. Corpus provenance is documented in `data/`.
