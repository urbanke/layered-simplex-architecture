# Offline ALT kernel-store construction

The builder creates a **new** store. It does not change `anchors_prod`, select a
production engine, change samples, or admit a numerical method. Construction,
integrity verification and numerical validation are separate steps.

The default plan covers depths 2–138 with a log-grid spacing of 0.02 and upper
bound 80. Every column is evaluated offline with the corrected contour/series
provider, `PMM_BUILD_EXACT=1`, oversampling 8, series tolerance `1e-13`, and contour
tail gap 40 nats. Its lower endpoint is
`floor((-60 - L*log(r+1)) / .02) * .02`. Depths 0 and 1 remain analytic in the
evaluator. No original column payload is copied or modified.

The anchor plan retains the original explicit positions, including every count
0–256, through eight anchors above the declared support maximum 1,045,889. With
the current source grid this gives 369 anchors per level, ending at 1,874,580:
50,553 columns and **11,633,407,536 payload bytes** for all 137 stored depths.
This is computed storage, not a measured construction-time or accuracy claim.

## Plan and local pilot

Use an explicit interpreter and set `PYTHONPATH` to this checkout's `src` when
it is not installed. Planning reads the original anchor metadata but performs no
kernel computation. Its destination must be new or empty.

```sh
export ALT_REPO=/absolute/path/to/clean/repository
export ALT_PYTHON=/absolute/path/to/alt-venv/bin/python
export PYTHONPATH="$ALT_REPO/src"
export ALT_KERNEL_STORE=/absolute/path/to/new-pilot-store

"$ALT_PYTHON" "$ALT_REPO/scripts/alt_build_kernel_store.py" plan \
  --anchors /absolute/path/to/anchors_prod/anchors.json \
  --levels 2,22,53,54,80,138 --out "$ALT_KERNEL_STORE"
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_build_kernel_store.py" build \
  --store "$ALT_KERNEL_STORE" --workers 2 --seal
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_build_kernel_store.py" verify \
  --store "$ALT_KERNEL_STORE"
```

The six-level pilot can be sealed because those six levels are its complete
plan. For the full store create a different directory and plan with
`--levels 2-138`. A build may select a subset with `--levels`; this does not alter
the plan. Sealing requires every planned level. A six-level sealed store cannot
later be extended in place. Alternatively, prepare the complete plan and build
six levels first without `--seal`; verify its checkpoints with
`verify --allow-incomplete` and resume its remaining levels later.

Run the declared sealed-store validation on the pilot before committing to the
full count-anchor grid. Off-grid interpolation and interpolation between count
anchors need separate checks. The earlier successful column and profile probes
do not certify the entire anchor grid. Use measured pilot wall time and peak
memory to choose cluster workers, memory and wall time.

## Checkpoints and immutability

`plan.json` freezes the geometry, exact anchor lists, support, numerical settings
and SHA256 identities of the builder, corrected numerical provider and private
runtime settings module. Other project files can change during development
without invalidating a construction checkpoint; a production cluster launch
additionally requires a clean, explicitly pinned repository commit.

Each worker owns a complete depth and writes a unique directory under
`attempts/`. Binary payloads contain concatenated little-endian float64 columns;
indexes carry exact lengths, starts, offsets and per-column hashes. A worker
verifies its finished payload and publishes `level_NNN.complete.json` last.
Resume accepts only completed levels with matching plan/source/settings and
verified payloads. Failed attempts and incomplete publications are preserved.
An interrupted incomplete level is recomputed in a new attempt; completed levels
are never overwritten. Per-level locks protect surviving workers even after a
coordinator disappears. If such a worker still holds its lock, wait for it to
finish or deliberately stop the old job before restarting; do not delete locks
to evade a live worker.

`manifest.json` seals the complete store and binds `plan.json`, every binary,
every index and every completion record. Engine configurations must pin all these
manifest entries **plus the manifest itself**. Integrity verification reads and
hashes the payload; run it on an allocated compute node for a full store.
Sealing is not numerical admission. Separate calibration and combined coverage
still determine whether an engine may run production experiments.

## SCITAS wrapper

`scripts/alt_build_kernel_store_slurm.sh` is a manual submission wrapper. It
uses the existing Jed account `lthc`, partition `standard`, QOS `serial`, one node
and one task. It does not submit another job, install dependencies or load
modules. Supply CPU, memory, wall-time and log settings explicitly using measured
pilot requirements. It caps processes at the allocated CPUs and 72, and forces
BLAS/OpenMP/NumExpr threads to one. Full levels are independent worker tasks;
this is one process pool, not a multi-node or job-array build.

Use an audited clean checkout and an existing separate Linux ALT environment.
The documented environment is
`/home/urbanke/venvs/alt2027-py313-20261007/bin/python`; verify that it remains the
intended environment before use. Do not copy a macOS virtualenv to Linux or
alter the existing PMWM environment. Load the same explicit modules used to
create the selected virtualenv before submission.

Prepare the full plan in a new scratch directory. Transfer the frozen plan as a
regular file if it was prepared elsewhere; source hashes are portable across
paths. Keep the build's plan and source unchanged. Set:

```sh
export ALT_REPO=/absolute/path/to/clean/cluster-checkout
export ALT_EXPECTED_COMMIT=FULL_40_CHARACTER_COMMIT
export ALT_PYTHON=/absolute/path/to/isolated-alt-venv/bin/python
export ALT_KERNEL_STORE=/scratch/urbanke/new-full-kernel-store
export ALT_KERNEL_PLAN_SHA256=FULL_64_CHARACTER_PLAN_FILE_SHA256
export ALT_BUILD_WORKERS=MEASURED_PROCESS_COUNT
export ALT_BUILD_SEAL=1
```

`ALT_BUILD_LEVELS` optionally selects a subset from the existing plan. Leave it
unset for all pending levels. `ALT_BUILD_SEAL=0` skips sealing after construction;
`1` requests it and fails if the plan is incomplete. These variables never alter
the planned grid or supported model domain.

Submit only after replacing resource placeholders with measured values and
creating the log directory:

```sh
sbatch --cpus-per-task="$ALT_BUILD_WORKERS" \
  --mem=MEASURED_MEMORY --time=MEASURED_WALL_TIME \
  --output=/scratch/urbanke/alt2027-logs/kernel-store-%j.out \
  "$ALT_REPO/scripts/alt_build_kernel_store_slurm.sh"
```

The wrapper enters an `srun` step, verifies actual compute-node membership,
checks the exact clean commit, explicit plan hash and numerical source hashes,
and records command/environment/scheduler provenance in a unique
`STORE/launches/UUID/` directory. It checks source and plan again on exit.
Builder progress and errors go to the Slurm log. In `exit.json`, `command_seconds`
measures the builder subprocess, including completed-level verification and
optional sealing; `seconds` additionally includes the final source/plan checks.
CPU counters subtract the preflight usage snapshot. Per-level records separately
report elapsed and kernel-construction times. The child RSS maximum is a lifetime
value that also includes preflight; it is not simultaneous aggregate worker
memory. Use Slurm job
accounting and a resource pilot to size the allocation. A missing exit record is
an interrupted or unconfirmed launch, not success.

Restart with the same frozen plan/store/commit after an interruption; verified
completed levels are skipped. Never use two simultaneous coordinators or a Slurm
array against one store. Scratch is temporary: archive the sealed store, its
hashes, source/environment identities and passed numerical reports to durable
storage before final paper results depend on it.

## Select the compiled interpolator

Build the native interpolation library explicitly on the execution platform,
from the final clean checkout. Keep the library, its adjacent `.json` build
manifest and engine configuration outside Git. Choose new output paths.
Prediction loads the selected library and verifies its identity; compilation is
an explicit preparation step.

```sh
export PYTHONPATH="$ALT_REPO/src"
export ALT_NATIVE_BINARY=/absolute/path/to/new-native-interpolator.so
export ALT_STORE_SPEC=/absolute/path/to/pinned-sealed-store-spec.json
export ALT_ENGINE_CONFIG=/absolute/path/to/new-sealed-engine.json

"$ALT_PYTHON" "$ALT_REPO/scripts/alt_build_sealed_native.py" \
  --out "$ALT_NATIVE_BINARY" --cc cc
```

Set `ALT_NATIVE_SHA256` to the `sha256` printed by this command, then configure
the engine:

```sh
export ALT_NATIVE_SHA256=EXACT_64_CHARACTER_BINARY_SHA256
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_experiments.py" --repo "$ALT_REPO" \
  configure-engine --store-directory "$ALT_KERNEL_STORE" \
  --store-spec "$ALT_STORE_SPEC" \
  --native-library-path "$ALT_NATIVE_BINARY" \
  --native-library-sha256 "$ALT_NATIVE_SHA256" \
  --out "$ALT_ENGINE_CONFIG"
```

The store specification supplies `files_sha256` and `settings`, including
`format: "sealed"`, declared domain limits and `saddle_min_depth: null`. Its
hash mapping covers the sealed manifest itself and every file bound by that
manifest. Both native command options are required together. They select
`interpolation_backend: "native"`; the library path, binary hash, C-source hash,
build manifest, compiler and floating-point flags enter the recorded identity.
Configuration opens and verifies the selected store and library. It does not
provide a numerical admission certificate.

## Validate the selected engine on SCITAS

`cluster/alt2027/sealed-validation.sbatch` runs one explicit mode in one node
and one Slurm task. It uses account `lthc`, partition `standard`, QOS `serial`,
and one thread per numerical library. Set these common variables:

```sh
export ALT_REPO=/absolute/path/to/final-clean-checkout
export ALT_EXPECTED_COMMIT=FULL_40_CHARACTER_COMMIT
export ALT_PYTHON=/absolute/path/to/Linux-virtualenv/bin/python
export ALT_ENGINE_CONFIG=/absolute/path/to/selected-sealed-engine.json
export ALT_VALIDATION_MODE=sealed_store
export ALT_WORKERS=MEASURED_PROCESS_COUNT
export ALT_OUT=/absolute/path/to/new-validation-run
```

Keep the virtualenv interpreter path as written, including its symlink into
the environment. `ALT_OUT` and its sibling `ALT_OUT.launch` must both be new
and outside the checkout. The wrapper verifies the clean commit, interpreter,
allocated node, CPU allowance and input identities; it checks source and inputs
again after execution.

| Mode | Work performed | `ALT_WORKERS` |
| --- | --- | --- |
| `setup` | Build a new native library and configure a new sealed engine | 1 |
| `sealed_store` | Declared kernel/interpolation checks for the sealed store | 1–allocated CPUs |
| `sealed_profile` | Shared-profile comparison with an explicit legacy engine | 1 |
| `kernel` | Independent kernel and special-function calibration | 1 |
| `prior` | Prior identities and independent integration checks | 1 |
| `power` | Powered-family calibration | 1–allocated CPUs |
| `depth` | Depth-family calibration and required refinement checks | 1–allocated CPUs |
| `chain` | Declared Bible predictive-chain checks | 1 |
| `regressions` | Full pytest suite and full Appendix checks | 1 |

All process counts are capped at 72 and at the allocated CPUs. Choose memory
and wall time from measured requirements; engine verification reads the full
store. Set `ALT_ALLOCATED_CPUS` to at least `ALT_WORKERS`. On the observed standard
partition, requested memory is limited to 7,000 MB per allocated CPU; a serial
20G validation job therefore reserves four CPUs while retaining one worker.
Supply the resource and log options explicitly:

```sh
sbatch --cpus-per-task="$ALT_ALLOCATED_CPUS" \
  --mem=MEASURED_MEMORY --time=MEASURED_WALL_TIME \
  --output=/absolute/path/to/logs/sealed-check-%j.out \
  "$ALT_REPO/cluster/alt2027/sealed-validation.sbatch"
```

`setup` is the wrapper alternative to the manual build/configure commands.
It additionally requires `ALT_KERNEL_STORE`, `ALT_STORE_SPEC` and a new
`ALT_NATIVE_BINARY`; `ALT_NATIVE_CC` optionally selects the compiler. Its
`ALT_ENGINE_CONFIG` must be new. Its result records engine setup.
`sealed_profile` additionally requires `ALT_LEGACY_ENGINE_CONFIG`, pointing
to the pinned reference engine. The seven numerical modes use their committed
specifications under `experiments/alt2027/`; `ALT_VALIDATION_CONFIG` can select
an explicit specification. Leave that variable unset for `setup` and
`regressions`.

## Assemble numerical admission

Finalize and commit the source first. Build/configure the selected library,
then complete the eight validation modes using the same finalized source,
selected engine and execution environment. Preserve the identical selected
binary and adjacent build manifest for subsequent execution. Platform-specific
native builds have separate identities and require matching evidence.

Create an evidence index with complete immutable Run directories for `kernel`,
`prior`, `power`, `depth`, `chain`, `sealed_store`, `sealed_profile` and
`regressions`. Retain failed runs separately. Use the existing assembly command
with the frozen protocol and selected engine:

```sh
"$ALT_PYTHON" "$ALT_REPO/scripts/alt_admission.py" assemble \
  --repo "$ALT_REPO" \
  --protocol "$ALT_REPO/experiments/alt2027/protocol.json" \
  --engine-config "$ALT_ENGINE_CONFIG" \
  --evidence /absolute/path/to/evidence-index.json \
  --batch-size 20 --out /absolute/path/to/new-admission
```

Inspect `assessment.json`; a passed assembly writes `calibration.json`.
Use that certificate, the same engine and matching batch settings in campaign
preparation and launch, following `cluster/alt2027/PRODUCTION.md`. Store sealing,
engine setup and numerical admission each retain their own records.
