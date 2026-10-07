# Assemble and launch the frozen ALT campaign

`scripts/alt_admission.py` assembles a certificate from the five existing
numerical suites and the full regression checks. It never changes a tolerance,
adds a numerical case, or turns a failed result into a pass. It writes
`calibration.json` only when every gate passes. An incomplete invocation writes
`assessment.json` with the missing or failing gates and exits with status 2
(pending) or 1 (failed); passed admission exits with status 0.

## Final source and evidence

Finish numerical fixes and commit the final code, tests, orchestration and
protocol. `experiments/alt2027/protocol.json` must have status `frozen`; its other
settings must remain the agreed revision. Keep the exact source checkout clean.
No production execution is needed to assemble its certificate.

Prepare an evidence index outside the checkout, with one path to each complete,
immutable `Run` directory. Each directory must include `manifest.json`,
`protocol.json`, `result.json` and all output files listed in its manifest. Paths
may be absolute, or relative to the index file. Point to the corrected power run;
retain the original failed run separately as a diagnostic record.

```json
{
  "kernel": "/absolute/path/to/completed-kernel-run",
  "prior": "/absolute/path/to/completed-prior-run",
  "depth": "/absolute/path/to/completed-depth-run",
  "power": "/absolute/path/to/completed-corrected-power-run",
  "chain": "/absolute/path/to/completed-chain-run",
  "regressions": "/absolute/path/to/final-linux-regression-run"
}
```

Missing entries are pending. Existing failed, corrupt or incompatible entries
are failed. An archived manifest alone is insufficient: unpack the complete
verified run so every recorded output can be checked. Assembly can run on another
machine from the same final source; it derives the campaign runtime from the
verified chain engine and requires every supplied suite to match it. Thus a
macOS regression run cannot admit a Linux campaign.

The numerical gates use the already committed suite specifications. Worker
allocation and host-local kernel-store paths are the only configuration fields
normalized for portability. The depth result must include its completed
assessment of every declared case, including required actual grid halving.
Kernel rows and special-function checks, the complete prior suite, all power
profiles and the declared Bible chain prefix must have passed their existing
criteria. The production engine must identify the committed store, match the
validated depth settings and use the validated nominal power settings.

## Reuse from an earlier clean source

The certificate records every changed source path and the specific reuse rule.
All unlisted changes require the affected suite to be rerun. The allowed changes
are deliberately narrow:

- Documentation and manuscript files do not invalidate numerical or regression
  evidence.
- Admission orchestration, launcher and test changes do not invalidate a
  numerical suite. They require fresh regression evidence.
- A change to `powers.py` or `power_validation.py` invalidates power and regression
  evidence. Kernel, prior, depth and chain evidence can be reused when every other
  scientific input remains identical.
- Changing the production protocol from `implementation` to `frozen` is allowed
  only when the entire remaining JSON object is identical to its recorded Git
  source. Changed trials, seeds, models or other settings are not a status freeze.

Numerical library files, corpus bytes, validation cases, store specifications and
locked dependencies have no blanket exception. The admission certificate binds
the final whole source-tree hash and frozen protocol even when it records reuse
of unaffected evidence from older commits. This preserves the existing strict
production runner gate without needlessly rerunning unchanged numerical suites.

## Record the final platform regressions

From the clean final checkout and pinned Linux virtualenv, execute this inside
an allocated compute-node step:

```sh
export PYTHONPATH="$ALT_REPO/src"
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_admission.py" regressions \
  --repo "$ALT_REPO" --out /absolute/path/to/new-linux-regression-run
```

The command runs the existing full `python -m pytest -q` and full
`scripts/validate_appendix_c.py`, saves both commands, exit codes and logs, and
records source/runtime identities in an immutable validation Run. It uses one
thread per numerical library. It does not use the quick appendix option. A
failed command leaves a failed Run; correct the issue and use a new directory.

## Assemble the certificate and plan

Keep all input files immutable. Use the exact same batch size in admission and
distributed plan creation; 20 is the existing default. Set `ALT_ENGINE_CONFIG` to
the explicit host-local configuration containing the complete pinned store
identity. Optional custom power settings must be supplied consistently to
admission, plan preparation and the launcher.

```sh
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_admission.py" assemble \
  --repo "$ALT_REPO" \
  --protocol "$ALT_REPO/experiments/alt2027/protocol.json" \
  --engine-config "$ALT_ENGINE_CONFIG" \
  --evidence /absolute/path/to/evidence-index.json \
  --batch-size 20 --out /absolute/path/to/new-admission
```

After this command passes, inspect `assessment.json`. It contains the source
reuse decisions, verified evidence hashes, runtime and every admission gate.
Then prepare the production plan:

```sh
export ALT_CALIBRATION=/absolute/path/to/new-admission/calibration.json
export ALT_PLAN=/absolute/path/to/new-production-plan.json
export ALT_OUT=/scratch/urbanke/alt2027-production-001
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_distributed.py" prepare \
  --repo "$ALT_REPO" \
  --protocol "$ALT_REPO/experiments/alt2027/protocol.json" \
  --purpose production --engine-config "$ALT_ENGINE_CONFIG" \
  --block-size 20 --factorial-block-size 500 --batch-size 20 \
  --out "$ALT_PLAN"
```

Twenty-trial blocks keep the longest measured power work below the scheduler
wall-time limit with room for its paired depth comparisons. They create 3,904
jobs with unchanged global trials and a 500-profile factorial block. The measured
throughput pilot determines final node count and resource requests.

Only after admission passes should the production array be submitted using
[the launcher guide](README.md). Preserve its explicit source SHA, interpreter,
engine, calibration, plan and shared output root. Array elements use 72 workers
per allocated node; the laptop launcher remains capped at 10. The certificate
covers the fixed experimental configuration, including batch implementation and
power settings. Different runtime environments require separate admitted
campaigns.

## Verify, merge and report

When every assigned job is complete, reconstruct the ordinary campaign layout:

```sh
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_distributed.py" merge \
  --repo "$ALT_REPO" --plan "$ALT_PLAN" --runs "$ALT_OUT" \
  --out /absolute/path/to/new-merged-campaign
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_experiments.py" report \
  --protocol "$ALT_REPO/experiments/alt2027/protocol.json" --purpose production \
  --sources /absolute/path/to/new-merged-campaign \
  --out /absolute/path/to/new-paper-report
```

Merge checks complete expected jobs, sample/trial identities, engines, runtime
and the retained admission record. Preserve failed attempts, original evidence
and source capsules. Archive raw campaign records and generated report assets
with durable locations and checksums before finalizing paper results. A passed
certificate covers the declared finite numerical checks; it does not assert a
uniform mathematical error bound over every possible sample.
