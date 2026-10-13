#!/usr/bin/env python3
"""Dependency-light integrity checks for the public GitHub repository."""
from __future__ import annotations

import argparse
import csv
import json
import math
import py_compile
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def passed(name: str, detail: str) -> Check:
    return Check(name, "PASS", detail)


def failed(name: str, detail: str) -> Check:
    return Check(name, "FAIL", detail)


def check_required_files(root: Path) -> Check:
    required = [
        "README.md",
        "LICENSE",
        "NOTICE",
        ".gitignore",
        ".github/workflows/verify.yml",
        "data/README.md",
        "docs/CAGF.md",
        "docs/PROJECT_CASE_STUDY.md",
        "docs/EVIDENCE_AND_ATTRIBUTION.md",
        "docs/OFFICIAL_TASKS_AND_COMPLIANCE.md",
        "src/Transformer_DCN.py",
        "src/embedding/ablation_presets.py",
        "src/embedding/confidence_features.py",
        "src/embedding/confidence_gated_fusion.py",
        "scripts/check_compliance.py",
        "results/ablation_summary.csv",
        "results/cagf_A5_downstream.csv",
        "results/cagf_A5_summary.json",
        "evidence/CAGF_DIAGNOSTICS.md",
        "evidence/codabench_result.png",
    ]
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        return failed("required_files", "missing: " + ", ".join(missing))
    return passed("required_files", f"{len(required)} public files found")


def check_python_syntax(root: Path) -> Check:
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "scripts").glob("*.py"))
    paths += [root / name for name in ("prediction.py", "run_expid.py", "run_param_tuner.py")]
    errors = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{path.relative_to(root)}: {exc.msg}")
    if errors:
        return failed("python_syntax", "; ".join(errors))
    return passed("python_syntax", f"compiled {len(paths)} Python files")


def check_ablation_results(path: Path) -> Check:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        variants = {row["variant_id"]: float(row["best_auc"]) for row in rows}
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        return failed("ablation_results", str(exc))
    expected = {"A0_baseline", "A1_rq", "A2_svd", "A3_mm", "A4_full", "A5_cagf"}
    if len(rows) != 6 or set(variants) != expected:
        return failed("ablation_results", "expected exactly the recorded A0-A5 variants")
    lift = variants["A1_rq"] - variants["A0_baseline"]
    gate_lift = variants["A5_cagf"] - variants["A4_full"]
    if not math.isclose(lift, 0.028603, abs_tol=1e-6):
        return failed("ablation_results", f"unexpected A1-A0 lift: {lift:.6f}")
    if not math.isclose(gate_lift, 0.001989, abs_tol=1e-6):
        return failed("ablation_results", f"unexpected A5-A4 lift: {gate_lift:.6f}")
    return passed(
        "ablation_results",
        f"A0-A5 retained; A1-A0={lift:+.6f}; A5-A4={gate_lift:+.6f}",
    )


def check_cagf_artifacts(root: Path) -> Check:
    try:
        summary = json.loads((root / "results/cagf_A5_summary.json").read_text(encoding="utf-8"))
        with (root / "results/cagf_A5_downstream.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            downstream = next(csv.DictReader(handle))
        num_items = int(summary["num_items"])
        stds = {name: float(stats["std"]) for name, stats in summary["weight_summary"].items()}
        auc = float(downstream["AUC"])
    except (OSError, ValueError, KeyError, StopIteration, TypeError, json.JSONDecodeError) as exc:
        return failed("cagf_artifacts", str(exc))
    if num_items != 91_718 or any(value <= 0 for value in stds.values()):
        return failed("cagf_artifacts", "item count or adaptive-weight diagnostics are inconsistent")
    if not math.isclose(auc, 0.891218, abs_tol=1e-6):
        return failed("cagf_artifacts", f"unexpected downstream AUC: {auc:.6f}")
    return passed("cagf_artifacts", f"{num_items:,} items; adaptive weights; AUC={auc:.6f}")


def check_attribution(root: Path) -> Check:
    paths = [root / "NOTICE", root / "docs/REFERENCES.md", root / "docs/EVIDENCE_AND_ATTRIBUTION.md"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths).lower()
    terms = ["fuxictr", "microlens", "transformer_dcn", "official baseline", "original"]
    missing = [term for term in terms if term not in text]
    if missing:
        return failed("attribution", "missing terms: " + ", ".join(missing))
    return passed("attribution", "baseline, dataset, reference ranker, and original work are separated")


def check_public_hygiene(root: Path) -> Check:
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".model", ".parquet", ".pkl", ".zip"}
    oversized = []
    forbidden = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            oversized.append(str(path.relative_to(root)))
        if path.suffix.lower() in forbidden_suffixes:
            forbidden.append(str(path.relative_to(root)))
    if oversized or forbidden:
        return failed("public_hygiene", f"oversized={oversized}; forbidden artifacts={forbidden}")
    return passed("public_hygiene", "no dataset, checkpoint, archive, or file larger than 10 MiB")


def check_common_secrets(root: Path) -> Check:
    pattern = re.compile(
        r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY|AKIA[0-9A-Z]{16}|"
        r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}"
    )
    hits = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(text):
            hits.append(str(path.relative_to(root)))
    if hits:
        return failed("secret_scan", "possible credential patterns in: " + ", ".join(hits))
    return passed("secret_scan", "no common credential patterns found")


def run_checks(root: Path = ROOT) -> list[Check]:
    return [
        check_required_files(root),
        check_python_syntax(root),
        check_ablation_results(root / "results/ablation_summary.csv"),
        check_cagf_artifacts(root),
        check_attribution(root),
        check_public_hygiene(root),
        check_common_secrets(root),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write a machine-readable report")
    args = parser.parse_args()
    checks = run_checks()
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")
    success = all(check.status == "PASS" for check in checks)
    if args.json:
        output = args.json if args.json.is_absolute() else ROOT / args.json
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"ok": success, "checks": [asdict(c) for c in checks]}, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"\n{'PROJECT READY' if success else 'PROJECT NEEDS ATTENTION'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
