from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "project_healthcheck", ROOT / "scripts/project_healthcheck.py"
)
healthcheck = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = healthcheck
SPEC.loader.exec_module(healthcheck)


class ProjectHealthcheckTests(unittest.TestCase):
    def test_public_repository_is_complete(self):
        result = healthcheck.check_required_files(ROOT)
        self.assertEqual(result.status, "PASS", result.detail)

    def test_recorded_ablation_arithmetic(self):
        result = healthcheck.check_ablation_results(ROOT / "results/ablation_summary.csv")
        self.assertEqual(result.status, "PASS", result.detail)
        self.assertIn("A5-A4=+0.001989", result.detail)

    def test_cagf_artifacts_match_reported_result(self):
        result = healthcheck.check_cagf_artifacts(ROOT)
        self.assertEqual(result.status, "PASS", result.detail)
        self.assertIn("AUC=0.891218", result.detail)

    def test_public_package_excludes_large_artifacts(self):
        result = healthcheck.check_public_hygiene(ROOT)
        self.assertEqual(result.status, "PASS", result.detail)


if __name__ == "__main__":
    unittest.main()
