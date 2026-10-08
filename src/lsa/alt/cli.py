"""Command line for the fixed-scope ALT campaign."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .artifacts import (
    Run,
    canonical_hash,
    read_json,
    sha256,
    source_identity,
    verify_run,
    write_json,
)


def plan(protocol):
    rows = []
    for name, config in protocol["experiments"].items():
        if name.startswith("benchmark_"):
            draws = len(config["targets"]) * len(config["n_values"]) * config["trials"]
        elif name == "spectrum":
            draws = len(config["panels"]) * len(config["alphas"]) * config["trials"]
        elif name == "factorial":
            draws = (
                len(config["ds"])
                * len(config["ns"])
                * len(config["alphas"])
                * config["trials"]
            )
        elif name == "depth_scaling":
            draws = len(config["alphas"]) * config["trials"]
        else:
            draws = None
        rows.append(
            {
                "experiment": name,
                "draws": draws,
                "artifact": config.get("artifact", config.get("artifacts")),
            }
        )
    return {
        "protocol_id": protocol["protocol_id"],
        "status": protocol["status"],
        "protocol_sha256": canonical_hash(protocol),
        "experiments": rows,
        "decisions": protocol.get("decisions", {}),
    }


def make_depth(args):
    from .depth import DepthEvaluator, StoreConfig

    options = (
        read_json(args.engine_config) if args.engine_config else {"mode": "reference"}
    )
    store = StoreConfig(**options["store"]) if options.get("store") else None
    calibration = read_json(args.calibration) if args.calibration else None
    return DepthEvaluator(
        mode=options["mode"],
        store=store,
        prediction_tolerance=options.get("prediction_tolerance", 1e-3),
        purpose="production" if args.purpose == "production" else "validation",
        calibration=calibration,
    )


def run_one(name, protocol, args, directory):
    config = protocol["experiments"][name]
    needs_depth = name not in ("architecture", "bible_secondary")
    evaluator = make_depth(args) if needs_depth else None
    power = None
    engine = {"depth": evaluator.configuration if evaluator else None}
    batch_size = getattr(args, "batch_size", None)
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch size must be positive")
    if batch_size is not None and name in (
        "benchmark_primary",
        "benchmark_powers",
        "spectrum",
        "factorial",
        "depth_scaling",
    ):
        engine["execution"] = {
            "batch_size": batch_size,
            "batch_implementation_sha256": sha256(
                Path(__file__).with_name("batch_depth.py")
            ),
        }
    if name == "benchmark_powers":
        from .powers import PowerEvaluator, PowerSettings

        power = PowerEvaluator(
            PowerSettings(
                **(read_json(args.power_settings) if args.power_settings else {})
            )
        )
        engine["power"] = {
            "settings": asdict(power.settings),
            "implementation_sha256": sha256(Path(__file__).with_name("powers.py")),
        }
    if args.purpose == "production":
        if not args.calibration:
            raise ValueError("production requires a recorded calibration")
        calibration = read_json(args.calibration)
        if (
            calibration.get("status") != "passed"
            or calibration.get("protocol_sha256") != canonical_hash(protocol)
            or name not in calibration.get("covered_experiments", [])
            or not calibration.get("required_checks_complete")
            or calibration.get("source_tree_sha256")
            != source_identity(args.repo)["tree_sha256"]
            or calibration.get("engine_sha256_by_experiment", {}).get(name)
            != canonical_hash(engine)
        ):
            raise ValueError("calibration does not cover this protocol and experiment")
    inputs = [args.protocol]
    for path in (args.engine_config, args.power_settings, args.calibration):
        if path:
            inputs.append(path)
    if name.startswith("bible"):
        inputs += [args.repo / config["corpus"], args.repo / config["manifest"]]
    try:
        with Run(
            directory,
            repo=args.repo,
            protocol=protocol,
            experiment=name,
            purpose=args.purpose,
            engine_identity=engine,
            inputs=inputs,
        ) as run:
            data = run.path / "data"
            if name.startswith("benchmark_"):
                from .benchmark import run_benchmark

                run_benchmark(
                    config,
                    data,
                    depth_evaluator=evaluator,
                    power_evaluator=power,
                    batch_size=batch_size,
                )
            elif name in ("spectrum", "factorial", "depth_scaling"):
                from .scaling import run_scaling

                run_scaling(
                    name, config, data, evaluator=evaluator, batch_size=batch_size
                )
            elif name.startswith("bible"):
                from .bible import run_bible

                run_bible(config, data, evaluator=evaluator, repo=args.repo)
            elif name == "architecture":
                from .architecture import run_architecture

                template = (
                    args.repo
                    / "manuscript/snapshots/2026-10-07-alt-start/figures/alt/construction_tikz.tex"
                )
                run_architecture(config, data, template_path=template)
            elif name == "validation":
                from .validation import run_validation

                run_validation(config, data, evaluator=evaluator)
            else:
                raise ValueError(f"unknown experiment {name}")
    finally:
        if evaluator:
            evaluator.close()
    verify_run(directory)


def report_campaign(protocol, args):
    # Every input is verified before rendering, including the saved common draws.
    names = list(protocol["experiments"])
    records = {name: verify_run(args.sources / name) for name in names}
    if (
        args.purpose == "production"
        and len({r["source"]["tree_sha256"] for r in records.values()}) != 1
    ):
        raise ValueError("production campaign combines different source trees")
    for name, record in records.items():
        if record["protocol_sha256"] != canonical_hash(protocol):
            raise ValueError(f"protocol differs in {name}")
        if record["purpose"] != args.purpose:
            raise ValueError(f"run purpose differs in {name}")
    inputs = [args.sources / name / "manifest.json" for name in names]
    with Run(
        args.out,
        repo=args.repo,
        protocol=protocol,
        experiment="reports",
        purpose=args.purpose,
        inputs=inputs,
    ) as run:
        from .benchmark_report import generate_reports
        from .bible import report_bible
        from .scaling import report_scaling

        sources = args.sources
        generate_reports(
            sources / "benchmark_primary/data",
            sources / "benchmark_powers/data",
            run.path / "benchmark",
            config=protocol["reports"],
        )
        for name in ("spectrum", "factorial", "depth_scaling"):
            report_scaling(sources / name / "data", run.path / name)
        primary = report_bible(sources / "bible/data", run.path / "bible")
        secondary = report_bible(
            sources / "bible_secondary/data", run.path / "bible_secondary"
        )
        from shutil import copyfile

        copyfile(
            sources / "architecture/data/construction_tikz.tex",
            run.path / "construction_tikz.tex",
        )
        controls = []
        # The full-corpus endpoints differ; interior checkpoints match.
        for left, right in zip(
            [r for r in primary["rows"] if r["n"] in primary["config"]["prefixes"]],
            [r for r in secondary["rows"] if r["n"] in secondary["config"]["prefixes"]],
        ):
            for method in ("add_one", "kt", "ristad", "good_turing_hybrid"):
                controls.append(
                    {
                        "primary_n": left["n"],
                        "secondary_n": right["n"],
                        "method": method,
                        "difference_bits": left["redundancy_bits"][method]
                        - right["redundancy_bits"][method],
                    }
                )
        write_json(run.path / "bible-tokenization-controls.json", controls)
        validation = read_json(sources / "validation/data/summary.json")
        write_json(run.path / "validation-status.json", validation)
        write_json(
            run.path / "source-runs.json",
            {
                name: {
                    "run_id": record["run_id"],
                    "manifest_sha256": sha256(sources / name / "manifest.json"),
                }
                for name, record in records.items()
            },
        )
    verify_run(args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser(
        "plan", help="show fixed grids, counts and protocol decisions"
    )
    inspect.add_argument("--protocol", type=Path, required=True)
    verify = sub.add_parser(
        "verify", help="check complete artifact inventory and hashes"
    )
    verify.add_argument("run", type=Path)
    from .calibration import SUITES

    calibrate = sub.add_parser("calibrate", help="run a declared numerical check suite")
    calibrate.add_argument("--suite", choices=SUITES, required=True)
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--out", type=Path, required=True)
    calibrate.add_argument("--engine-config", type=Path)
    calibrate.add_argument("--workers", type=int)
    configure = sub.add_parser(
        "configure-engine", help="bind the pinned store to a local path"
    )
    configure.add_argument("--store-directory", type=Path, required=True)
    configure.add_argument(
        "--store-spec", type=Path,
        help="explicit pinned store specification (default: legacy store-candidate.json)",
    )
    configure.add_argument("--out", type=Path, required=True)
    configure.add_argument("--native-library-path", type=Path)
    configure.add_argument("--native-library-sha256")
    assess = sub.add_parser(
        "assess-depth", help="check saved depth profiles and actual refinement"
    )
    assess.add_argument("--sources", type=Path, required=True)
    assess.add_argument("--out", type=Path, required=True)
    assess.add_argument("--engine-config", type=Path, required=True)
    assess.add_argument("--config", type=Path)
    assess.add_argument(
        "--supplement",
        action="store_true",
        help="evaluate finer grids where actual spacing was not halved",
    )
    for command in ("run", "campaign", "report"):
        p = sub.add_parser(command)
        p.add_argument("--protocol", type=Path, required=True)
        p.add_argument(
            "--purpose", choices=("smoke", "validation", "production"), required=True
        )
        p.add_argument("--out", type=Path, required=True)
        if command == "report":
            p.add_argument("--sources", type=Path, required=True)
        else:
            p.add_argument("--engine-config", type=Path)
            p.add_argument("--power-settings", type=Path)
            p.add_argument("--calibration", type=Path)
            p.add_argument(
                "--batch-size",
                type=int,
                default=20,
                help="maximum saved profiles sharing kernel preparation",
            )
        if command == "run":
            p.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    if args.command == "assess-depth":
        from .depth_validation_assessment import assess_depth_run

        result = assess_depth_run(
            args.sources,
            args.out,
            engine_config=read_json(args.engine_config),
            run_supplemental=args.supplement,
            config_path=args.config,
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "status",
                        "completed_cases_read",
                        "expected_cases",
                        "pending_case_ids",
                    )
                }
            )
        )
        return {"passed": 0, "failed": 1, "pending": 2}[result["status"]]
    if args.command == "configure-engine":
        from .depth import DepthEvaluator, StoreConfig

        spec = read_json(
            args.store_spec or args.repo / "experiments/alt2027/store-candidate.json"
        )
        options = {
            "mode": "store",
            "prediction_tolerance": 1e-3,
            "store": {
                "path": str(args.store_directory.resolve()),
                "files_sha256": spec["files_sha256"],
                **spec["settings"],
            },
        }
        if bool(args.native_library_path) != bool(args.native_library_sha256):
            raise ValueError("native interpolation requires both library path and SHA256")
        if args.native_library_path:
            options["store"].update(
                interpolation_backend="native",
                native_library_path=str(args.native_library_path.resolve()),
                native_library_sha256=args.native_library_sha256,
            )
        # Opening performs complete identity checks and permits no store writes.
        with DepthEvaluator(
            mode="store",
            store=StoreConfig(**options["store"]),
            prediction_tolerance=options["prediction_tolerance"],
        ) as evaluator:
            identity = evaluator.configuration_sha256
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, options)
        print(json.dumps({"configuration_sha256": identity, "path": str(args.out)}))
        return 0
    if args.command == "calibrate":
        from .calibration import run_suite

        result = run_suite(
            args.suite,
            args.config,
            args.out,
            repo=args.repo,
            engine_path=args.engine_config,
            workers=args.workers,
        )
        print(json.dumps({"suite": args.suite, "status": result["status"]}))
        return 0
    if args.command == "verify":
        record = verify_run(args.run)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "run_id": record["run_id"],
                    "purpose": record["purpose"],
                }
            )
        )
        return 0
    protocol = read_json(args.protocol)
    if args.command == "plan":
        print(json.dumps(plan(protocol), indent=2))
    elif args.command == "report":
        report_campaign(protocol, args)
    elif args.command == "run":
        if args.experiment not in protocol["experiments"]:
            parser.error("experiment is absent from protocol")
        run_one(args.experiment, protocol, args, args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        for name in protocol["experiments"]:
            print(f"Running {name}", flush=True)
            run_one(name, protocol, args, args.out / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
