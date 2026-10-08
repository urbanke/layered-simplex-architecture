# ALT laptop and Jed launchers

These launchers run an immutable distributed plan with an explicit checkout,
interpreter, numerical engine configuration and output directory. They preserve
scientific sample counts, seeds and grids. Parallel worker counts control
resources; they never change the experiment specification.

| File | Purpose |
|---|---|
| `laptop.sh` | One local controller, up to 10 processes |
| `jed-array.sbatch` | One controller and 72 processes per independent node |
| `jed-preflight.sbatch` | Complete store identity and tiny checks on a compute node |
| `run_worker.sh`, `launch.py` | Source, interpreter, allocation and input guards; immutable launch records |
| `preflight.py` | All 106 store files plus independent/reference comparisons at d=2, N=3, depths 0–2 |

## Known Jed setup

On 2026-10-07, access through `ssh jed` was restored after enabling the EPFL VPN.
The alias connects as `urbanke` to `jed.hpc.epfl.ch`. The checked account is `lthc`,
partition `standard`, with `debug`, `serial` and `parallel` QOS. Jed provides 72
cores per node. The array template uses `serial` because each array element is
one independent one-node job. It does not combine nodes into an MPI job.

The existing environment is `/home/urbanke/venvs/product-model-memory`, with
Python 3.13.15, NumPy 2.5.2, SciPy 1.18.0 and mpmath 1.4.1 at inspection time.
Keep it unchanged. Prepare a separate ALT virtualenv from an explicit Linux
interpreter and `requirements-alt.lock`; record its exact interpreter, installed
packages and modules. The current lock uses NumPy 2.5.1 and was tested on macOS
with Python 3.14.6, so the discovered environment is not the locked ALT environment
or an admitted Linux calibration. Never copy a macOS virtualenv onto Linux.
If the exact pins cannot be installed on the chosen Linux interpreter, resolve
that as an explicit environment revision before calibration.

The separate environment `/home/urbanke/venvs/alt2027-py313-20261007` was then
created with Python 3.13.15 and every package pinned by `requirements-alt.lock`.
Installation and `pip check` passed, including NumPy 2.5.1, SciPy 1.18.0 and
mpmath 1.4.1. Setup records are in `/scratch/urbanke/lsa-alt2027/setup`;
compute-node numerical checks determine platform admission.

The cluster store is
`/home/urbanke/Projects/product_model_with_memory/tables/anchors_prod`.
The inspected `anchors.json` and `manifest.json` hashes match the laptop:

- `anchors.json`: `5c409244487deeba8d597ad66fa13479a8d15476d55d4509c33b2ec797b07c68`
- `manifest.json`: `cf0449a4b704174890f0a1fb95f20954839dd7266ae307e799d1dc6d0234ffb5`

The compute-node preflight verifies **all 106 files, 2,802,407,082 bytes** against
`experiments/alt2027/store-candidate.json`. The two inspected hashes alone do not
substitute for this check. The adapter opens the store read-only and cannot build
or modify it.

## Freeze inputs before launch

Use an audited, clean source commit and the same checkout for the launcher and
worker. Supply the full 40-character SHA as `--expected-commit`. The plan must
contain matching `source_commit`, `source_tree_sha256`, `protocol`, and
`protocol_sha256` fields. Recreate the plan after any source changes, including
changes to these wrappers. Do not edit a running checkout or environment.

The launcher requires absolute paths for the repository, plan, output root,
virtualenv Python and engine configuration. Optional calibration and power
settings paths must also be absolute. A store path inside the engine configuration
must be absolute; only that machine-specific path changes when relocating the
same pinned store. Keep configuration and plan files immutable after creation.
Store and engine identity checks remain the controller's responsibility for
campaign work; the wrapper does not replace production admission.

Every production invocation requires `--calibration` and a frozen protocol. The
controller checks the certificate's source, protocol, runtime, engine and assigned
experiment coverage. A macOS certificate does not grant Linux admission. A
successful tiny preflight is not a production calibration.

Use a new output root for each independent campaign/preflight. For a coordinated
array, all elements use the same plan, output root and controller count. Outputs
inside a checkout must be in a Git-ignored directory such as `output/`; an
external output root is preferable for production. Keep smoke, validation and
production records separate.

The process environment caps BLAS/OpenMP/NumExpr threads at one, disables user
site packages, and supplies only this checkout's `src` on `PYTHONPATH`. Python
must belong to the explicitly supplied virtualenv. The launcher records all
installed package versions, Python build/compiler details, modules and selected
Slurm variables.

## Laptop

The positional convenience wrapper accepts:

```text
laptop.sh REPO_ABS EXPECTED_COMMIT PLAN_ABS OUT_ABS VENV_PYTHON_ABS ENGINE_CONFIG_ABS [WORKERS=10] [CALIBRATION_ABS] [POWER_SETTINGS_ABS]
```

For example, after replacing each placeholder:

```sh
bash /absolute/path/to/repository/cluster/alt2027/laptop.sh \
  /absolute/path/to/repository FULL_40_CHARACTER_COMMIT \
  /absolute/path/to/laptop-plan.json /absolute/path/to/new-output-root \
  /absolute/path/to/alt-venv/bin/python /absolute/path/to/engine.json \
  10 /absolute/path/to/laptop-calibration.json
```

The worker count must be 1–10. The controller index is 0 of 1. Named arguments
are available through `run_worker.sh`; they are preferable when supplying only
some optional inputs:

```sh
bash "$ALT_REPO/cluster/alt2027/run_worker.sh" \
  --repo "$ALT_REPO" --expected-commit "$ALT_EXPECTED_COMMIT" \
  --plan "$ALT_PLAN" --out "$ALT_OUT" --python "$ALT_PYTHON" \
  --engine-config "$ALT_ENGINE_CONFIG" --workers 10 \
  --worker-index 0 --worker-count 1 --host-profile laptop \
  --calibration "$ALT_CALIBRATION"
```

## Jed compute-node preflight

Prepare the isolated ALT environment and clean checkout first. Load the same
explicitly versioned modules used to build that environment before submission.
The wrappers never install packages or load default modules. Keep heavy hashing,
calibration and numerical execution on allocated compute nodes. In particular,
`alt_experiments.py configure-engine` hashes the whole store and belongs inside
an allocated step. Small configuration/plan preparation does not run experiments.

Set these values in the configured Jed shell; the checkout, ALT environment and
new run paths below are placeholders:

```sh
export ALT_REPO=/absolute/path/to/clean/repository
export ALT_EXPECTED_COMMIT=FULL_40_CHARACTER_COMMIT
export ALT_PYTHON=/absolute/path/to/isolated-alt-venv/bin/python
export ALT_ENGINE_CONFIG=/absolute/path/to/jed-engine.json
export ALT_PLAN=/absolute/path/to/jed-validation-plan.json
export ALT_OUT=/scratch/urbanke/alt2027-preflight-001
mkdir -p /scratch/urbanke/alt2027-logs
```

The engine configuration must carry the exact store hash mapping from the
committed candidate specification and the explicit cluster store path. Prepare a
validation plan with the distributed controller, after the source is committed:

```sh
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_distributed.py" prepare \
  --repo "$ALT_REPO" --protocol "$ALT_REPO/experiments/alt2027/smoke.json" \
  --purpose validation --engine-config "$ALT_ENGINE_CONFIG" --out "$ALT_PLAN" \
  --block-size 100 --factorial-block-size 500 --batch-size 20
sbatch "$ALT_REPO/cluster/alt2027/jed-preflight.sbatch"
```

This requests two CPUs, 8G and ten minutes in `debug` QOS. It verifies the source
and input identities, hashes all pinned store files through the read-only
adapter, then compares fixed tiny store/reference evidence and predictions. It
records the plan as validation context and **does not run its campaign jobs**.
Review the saved preflight result and exit record. Full domain/platform
calibration remains necessary before production.

The launcher requires an allocation and verifies that the current hostname is
one of its nodes. An interactive `salloc` shell that remains on a login node is
insufficient. Both templates execute the launcher inside an `srun` step.

## Jed production array

Set `ALT_PLAN` to the admitted production plan, `ALT_OUT` to its new shared output
root and `ALT_CALIBRATION` to the exact Linux certificate. Optional
`ALT_POWER_SETTINGS` is forwarded as `--power-settings`; prepare the plan with
the same settings. For example, four node controllers with at most two nodes
running simultaneously:

```sh
export ALT_PLAN=/absolute/path/to/jed-production-plan.json
export ALT_OUT=/scratch/urbanke/alt2027-production-001
export ALT_CALIBRATION=/absolute/path/to/jed-calibration.json
ALT_ARRAY_SIZE=4
ALT_CONCURRENT_NODES=2
sbatch --array="0-$((ALT_ARRAY_SIZE - 1))%${ALT_CONCURRENT_NODES}" \
  "$ALT_REPO/cluster/alt2027/jed-array.sbatch"
```

Use a contiguous zero-based array `0..K-1`. Its task ID becomes `--worker-index`
and K becomes `--worker-count`. Sparse arrays are rejected. The `%2` suffix
limits simultaneous nodes without changing assignments. Each node uses one
Python process pool of 72 workers, one Slurm task and 72 allocated CPUs.

The template requests 440G and 24 hours per node as explicit starting values.
Choose memory, wall time and concurrency from the measured pilot and current
account limits; these defaults are not measured requirements. Slurm command-line
options can override memory/time directives. The wrapper refuses more workers
than allocated CPUs or 72, and refuses production in `debug` QOS.

The output root must be shared across participating nodes; do not use node-local
`/tmp`. Treat scratch as temporary storage. Archive verified records and manifests
to durable storage before finalizing paper results.

## Records and restarts

Every invocation creates UUID-qualified immutable files in `OUT/launches/` for
source and input hashes, exact command, resource/environment provenance and exit
status. Preflight adds a separate measured result with all 106 store digests.
Inputs and source are checked again at exit. Missing exit metadata indicates an
interrupted or unconfirmed run, not success.

The controller owns completed `OUT/jobs/ID` records and retained failed attempts;
restart `work` with the same plan and output root to verify and skip completed
jobs. OS file locks prevent two live controllers from executing the same job;
an interrupted attempt remains under `OUT/attempts/` and a retry receives a new
directory. Stop all old controllers before changing their count K, then restart
the complete contiguous array. Assignments change while global trial IDs stay
fixed. Numerical failure stops new dispatch and retains the other in-flight
results; the scheduler wall-time bounds those evaluations. Use
`squeue -u urbanke` and `sacct -j JOB_ID` for scheduler status/resources. Preserve
failure records, retain the same K for simultaneously running controllers, and never combine
incompatible plans/platforms into one output root. These wrappers do not add
another automatic retry or result-merging layer.
