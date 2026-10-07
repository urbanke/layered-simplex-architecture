"""Fast launcher guards; no scheduler submission, store hashing or campaign runs."""

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import launch
from preflight import reference_preflight, store_preflight

from lsa.alt.artifacts import canonical_hash
from lsa.alt.distributed import engine_options, power_options


class LauncherGuards(unittest.TestCase):
    def setUp(self):
        self.source = {"commit": "a" * 40, "tree_sha256": "b" * 64, "dirty": False}
        protocol = {"status": "implementation", "experiments": {}}
        self.plan = {
            "schema_version": 1,
            "block_size": 100,
            "factorial_block_size": 500,
            "batch_size": 20,
            "jobs": [],
            "purpose": "validation",
            "protocol": protocol,
            "protocol_sha256": canonical_hash(protocol),
            "source_commit": self.source["commit"],
            "source_tree_sha256": self.source["tree_sha256"],
            "engine_options_sha256": canonical_hash(
                engine_options({"mode": "reference"})
            ),
            "power_settings_sha256": canonical_hash(power_options({})),
        }
        self.allocation = {
            "SLURM_JOB_ID": "123",
            "SLURM_JOB_NODELIST": "jed001",
            "SLURM_CPUS_PER_TASK": "72",
            "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NTASKS": "1",
        }

    def test_exact_plan(self):
        self.assertEqual(launch.check_plan(self.plan, self.source), "validation")

    def test_plan_binds_numerical_inputs(self):
        launch.check_numerical_inputs(self.plan, {"mode": "reference"}, {})
        with self.assertRaisesRegex(ValueError, "engine configuration"):
            launch.check_numerical_inputs(
                self.plan, {"mode": "reference", "prediction_tolerance": 1e-5}, {}
            )
        with self.assertRaisesRegex(ValueError, "power settings"):
            launch.check_numerical_inputs(
                self.plan, {"mode": "reference"}, {"kernel_tail_drop": 48.0}
            )

    def test_plan_rejects_source_and_protocol_changes(self):
        for field in ("source_commit", "source_tree_sha256", "protocol_sha256"):
            with self.subTest(field=field):
                changed = dict(self.plan, **{field: "wrong"})
                with self.assertRaises(ValueError):
                    launch.check_plan(changed, self.source)
        with self.assertRaises(ValueError):
            launch.check_plan(self.plan, dict(self.source, dirty=True))

    def test_production_requires_frozen_protocol(self):
        self.plan["purpose"] = "production"
        with self.assertRaisesRegex(ValueError, "frozen"):
            launch.check_plan(self.plan, self.source)
        self.plan["protocol"]["status"] = "frozen"
        self.plan["protocol_sha256"] = canonical_hash(self.plan["protocol"])
        self.assertEqual(launch.check_plan(self.plan, self.source), "production")

    @patch.dict(os.environ, {}, clear=True)
    @patch("launch.socket.gethostname", return_value="laptop.local")
    def test_laptop_cap(self, _hostname):
        args = SimpleNamespace(host_profile="laptop", workers=10)
        self.assertIsNone(launch.scheduler_check(args, "validation"))
        args.workers = 11
        with self.assertRaisesRegex(ValueError, "10 worker"):
            launch.scheduler_check(args, "validation")

    @patch.dict(os.environ, {}, clear=True)
    @patch("launch.socket.gethostname", return_value="jed-login")
    def test_laptop_profile_refuses_cluster(self, _hostname):
        with self.assertRaisesRegex(ValueError, "SCITAS profile"):
            launch.scheduler_check(
                SimpleNamespace(host_profile="laptop", workers=1), "validation"
            )

    @patch.dict(os.environ, {}, clear=True)
    def test_allocation_required(self):
        with self.assertRaisesRegex(ValueError, "allocation"):
            launch.scheduler_check(
                SimpleNamespace(host_profile="scitas", workers=1), "validation"
            )

    @patch("launch.socket.gethostname", return_value="jed001.hpc.epfl.ch")
    @patch("launch.checked", side_effect=["jed001", "JobId=123 QOS=serial"])
    def test_allocated_node_72_workers(self, _checked, _hostname):
        with patch.dict(os.environ, self.allocation, clear=True):
            result = launch.scheduler_check(
                SimpleNamespace(host_profile="scitas", workers=72), "production"
            )
        self.assertEqual(result["allocated_nodes"], ["jed001"])

    @patch("launch.socket.gethostname", return_value="jed-login")
    @patch("launch.checked", return_value="jed001")
    def test_salloc_login_shell_refused(self, _checked, _hostname):
        with (
            patch.dict(os.environ, self.allocation, clear=True),
            self.assertRaisesRegex(ValueError, "outside"),
        ):
            launch.scheduler_check(
                SimpleNamespace(host_profile="scitas", workers=1), "validation"
            )

    @patch("launch.socket.gethostname", return_value="jed001")
    @patch("launch.checked", return_value="jed001")
    def test_worker_cap_and_single_task(self, _checked, _hostname):
        cases = [
            (self.allocation, 73),
            (dict(self.allocation, SLURM_CPUS_PER_TASK="2"), 3),
            (dict(self.allocation, SLURM_NTASKS="2"), 1),
            (dict(self.allocation, SLURM_JOB_NUM_NODES="2"), 1),
        ]
        for allocation, workers in cases:
            with (
                self.subTest(allocation=allocation, workers=workers),
                patch.dict(os.environ, allocation, clear=True),
                self.assertRaises(ValueError),
            ):
                launch.scheduler_check(
                    SimpleNamespace(host_profile="scitas", workers=workers),
                    "validation",
                )

    @patch("launch.socket.gethostname", return_value="jed001")
    @patch("launch.checked", side_effect=["jed001", "JobId=123 QOS=debug"])
    def test_debug_production_refused(self, _checked, _hostname):
        with (
            patch.dict(os.environ, self.allocation, clear=True),
            self.assertRaisesRegex(ValueError, "non-debug"),
        ):
            launch.scheduler_check(
                SimpleNamespace(host_profile="scitas", workers=1), "production"
            )

    def test_store_preflight_rejects_unpinned_engine_before_opening(self):
        repo = Path(__file__).resolve().parents[2]
        with patch("lsa.alt.depth.DepthEvaluator") as evaluator:
            with self.assertRaisesRegex(ValueError, "106-file store"):
                store_preflight(repo, {"mode": "reference"}, {})
            evaluator.assert_not_called()

    def test_tiny_independent_reference(self):
        self.assertEqual(reference_preflight()["status"], "passed")


if __name__ == "__main__":
    unittest.main()
