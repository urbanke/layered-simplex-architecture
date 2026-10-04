# A Layered Simplex Architecture for Large Alphabets — code

**Paper:** *A Layered Simplex Architecture for Large Alphabets*,
Meir Feder, Yaniv Fogel, Ruediger Urbanke.
arXiv: [2608.19908](https://arxiv.org/abs/2608.19908).

This repository contains everything needed to reproduce the numerical
results of the paper: the exact evaluation machinery for the layered
simplex architecture (LSA) prior, the classical estimators it is compared
against, one script per figure and table, and the validation checks of
Appendix C.

If you are an LLM or an automated agent: read [`AGENTS.md`](AGENTS.md)
first. It maps every result of the paper to the command that reproduces
it, with expected values and runtimes. The same map is machine-readable
in [`results_manifest.json`](results_manifest.json), and the paper's
numbers are in [`expected/paper_values.json`](expected/paper_values.json).

## Setup

Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Verify the installation (about a minute):

```bash
python scripts/validate_appendix_c.py --quick
```

This runs the implementation checks of Appendix C at reduced scale: the
layer recursion against independent quadrature, the closed forms at
L = 1 (Proposition 1), the exchangeability identities, predictive
normalization, and the profile-weight identities of Appendix A. All
must pass. `python -m pytest -q` runs the full test suite (~10 min).

## Reproducing the results

Every experiment writes its results as JSON plus rendered figures and
`.tsv` tables. Every experiment has a fast smoke variant (same code
path, reduced scale) listed in `results_manifest.json`. Fixed seeds are
the defaults, so runs are repeatable draw for draw.

| Result | Command(s) | Time (8 cores) |
|---|---|---|
| Figure 1 (a draw of the prior) | `python scripts/fig1_prior_draws.py --d 24 --l 4 --out output/fig1` | seconds |
| Figure 2, Table 1 (depth tilts the spectrum) | `depth_tilt_experiment.py` for d = 10^3, 10^4, 10^6, then `depth_tilt_report.py` | hours (d = 10^6 dominates) |
| Figures 3–4, Tables 2–3 (scaling laws) | `factorial_scaling_experiment.py`, then `scaling_report.py` | hours |
| Figure 5, Table 4 (depth scaling) | `depth_scaling_experiment.py` | 1–2 h |
| Figure 6, Tables 5–6 (competitive benchmark) | `benchmark_experiment.py`, then `benchmark_report.py` | hours |
| Table 7, Figure 7 (the Bible) | `get_kjv.py`, `unigram_experiment.py`, `bible_baselines_experiment.py`, `bible_report.py` | ~1 h |
| Table 8 (order-one Bible models) | `state_family_experiment.py` (two runs) | heavy |
| §5.3 byte-pair check (in text) | `bible_bpe_check.py` (three vocabulary sizes) | minutes each |
| §5.4 online context code (in text) | `bible_online_states_experiment.py` | ~1 h |
| Appendix C (validation) | `validate_appendix_c.py` and `pytest` | ~25 min |

Exact command lines with all arguments are in `results_manifest.json`;
`bash reproduce.sh --smoke` runs every smoke variant in sequence
(roughly 20 minutes), and `bash reproduce.sh --dry-run` prints the full
plan without running anything.

A word on the corpus: `scripts/get_kjv.py` builds the King James Bible
corpus (Project Gutenberg eBook #10, verse numbers removed) and checks
its token statistics. A compressed copy of the tokenized corpus ships in
`data/kjv.txt.gz`, so no download is needed and the edition is pinned.
This is exactly the paper's corpus — 915,860 tokens, 13,550 distinct
types — and the script asserts those counts when it runs.

## Layout

```
src/lsa/            the package
  product_simplex.py    sampling the LSA prior (Section 3, Figure 1)
  mixture_weights.py    reference numerics for the mixture weights q_lambda (Appendix B)
  pattern_weights.py    profile weights A_lambda (Appendix A)
  layered.py            production numerics: tables, global-peak scan (Appendix B)
  fast_tables.py        batched, disk-cached moment tables (Appendix B.1-B.2)
  mellin.py             exact kernel rows via Mellin–Barnes contours (Appendix B.3)
  universal_tables.py   designed anchor stores for heavy counts (Appendix B.2)
  codelength.py         exact codelength of the depth-averaged predictor (Section 5.3)
  corpus.py, pairs.py   corpus loading, counts, vocabulary reduction
  state_family.py       per-state predictors over nested state maps (Section 5.4)
  estimators.py         add-one, KT, Braess–Sauer, Good–Turing, Ristad, oracle,
                        and the exact LSA predictives (Section 5.1)
scripts/            one experiment or report per file (see results_manifest.json)
tests/              the test suite; includes cross-validation of the two
                    numerics implementations against each other
expected/           the paper's numbers, for automated comparison
data/               corpora (kjv.txt.gz ships; everything else is rebuilt)
```

Two implementation notes. First, the package contains two independent
implementations of the Appendix-B numerics — `mixture_weights.py` (the
original reference) and `layered.py` with its table machinery (the
production path) — and the test suite holds them to each other; all
corpus experiments use the production path. Second, moment-table rows
persist in a certified store at `tables/universal_v2` (gitignored;
created on first use). Rows are built once and reused by every later
experiment, which makes repeat runs much faster; the store is safe to
delete and grows to a few GB across the full reproduction.

## License and citation

MIT license (see `LICENSE`). To cite this code or the paper, see
`CITATION.cff` (GitHub's "Cite this repository" button uses it); the
arXiv reference will be added there as soon as the preprint is up.

## Revised submission Tables 1 and 2

The submission numbers the competitive benchmark as **Table 1** and the Bible
compression experiment as **Table 2** (Tables 5 and 7 in the preprint mapping
above). Reproduce the requested revisions with:

```bash
python scripts/submission_tables.py --smoke --jobs 2
python scripts/submission_tables.py --table 1 --jobs 8
python scripts/submission_tables.py --table 2 --jobs 8
```

The full Table 1 run uses `d=10000`, `n=1000`, 20 paired trials, the original
seed/target generation, and equal prior weights over depths **0 through 80**.
The full Table 2 run retains the pinned 915,860-token Bible corpus, `d=100000`,
and all five original checkpoints; its depth ceiling follows the original
Bible rule `round(2 c* ln d)`, now including depth zero. Both reports include
AD and Dir-tau. Outputs are under `output/submission_tables/table{1,2}`,
including `table1.tsv` / `table2.tsv`, JSON results, and the existing report
figures. Smoke mode uses one trial/two targets for Table 1 and only the first
Bible checkpoint for Table 2; its numbers are not full-run results.

The Table 1 driver enables `--transient-cache --checkpoint`. Depth 80 exceeds
the universal table store's depth ceiling, so the original engine falls back
to profile-dependent recursion tables. Keeping every profile's cache can use
tens of GB. The revised driver removes each sample's temporary tables after
evaluation and atomically saves its losses and posterior in `table1/cells/`.
Rerunning the same command resumes completed cells with the same random draws;
changed experiment settings are rejected for an existing checkpoint directory.
Final `results.json` is written only after every cell completes. This changes
storage and restart behavior, not the estimator or numerical resolution.

`L=0` is the fixed uniform distribution, with sequence evidence `d**(-n)`
and predictive probability `1/d`. Its prior weight equals that of each other
depth. The posterior weights are evidence-weighted, not equal at prediction
time. `--include-zero` on `benchmark_experiment.py` or `unigram_experiment.py`
enables it explicitly; without the flag their original behavior is retained.
Result JSON records the actual depth labels, and the reports use those labels.

`--extra-baselines` enables AD and Dir-tau on the benchmark and Bible baseline
scripts. **Dir-tau** is a symmetric Dirichlet concentration mixture, with equal
prior weights on the 29 per-coordinate concentrations `beta=2**j`,
`j=-24,...,4`. Its corpus score is the sequence marginal likelihood, not a
predictive score fitted to the same prefix. Mixtures over symmetric Dirichlet
concentration have precedents in [Nemenman, Shafee and Bialek (2001)](
https://proceedings.neurips.cc/paper_files/paper/2001/file/d46e1fcf4c07ce4a69ee07e4134bcef1-Paper.pdf);
this grid and predictive baseline are not their NSB entropy estimator.

**AD** uses the unregularized discount `c1/(c1+2*c2)` and spreads freed mass
uniformly over unseen symbols, with a uniform prediction at empty history.
See [Chen and Goodman (1999), Section 2.6](
https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf) for this
discount estimate and its attribution to Ney, Essen and Kneser (1994).
No clipping or fallback is introduced: a zero-probability event yields
`infinite`, and an undefined rule yields `undefined`. JSON stores null numeric
values with explicit diagnostics instead of nonstandard Infinity/NaN literals.
In particular, this AD rule assigns zero probability at token 9 of the pinned
Bible, so all its reported prefix code lengths are infinite. This is a property
of the specified rule, not a claim about regularized AD variants.
