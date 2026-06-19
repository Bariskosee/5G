"""Static Docker packaging verification — no Docker daemon required.

Checks that Dockerfile, .dockerignore, and local model weights are in the
expected state before running docker build. Prints PASS/FAIL for each check.

Usage
-----
python scripts/check_docker_packaging.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_BASE_IMAGE = "nvidia/cuda:12.1.0-base-ubuntu22.04"
_CMD_FORMS = ('["python3", "main.py"]', "python3 main.py")
MODEL_PATH = REPO_ROOT / "models" / "model_b_plate" / "best.pt"


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    return passed


def _dockerfile_content() -> str:
    path = REPO_ROOT / "Dockerfile"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _dockerignore_content() -> str:
    path = REPO_ROOT / ".dockerignore"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _gitignore_ignores_model() -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(MODEL_PATH)],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    return result.returncode == 0


def main() -> int:
    print("Docker Packaging Checks")
    print("=" * 40)

    results: list[bool] = []
    df = _dockerfile_content()
    di = _dockerignore_content()

    results.append(_check(
        "Dockerfile exists",
        bool(df),
        "Dockerfile" if df else "MISSING",
    ))

    results.append(_check(
        f"Dockerfile FROM: {EXPECTED_BASE_IMAGE}",
        EXPECTED_BASE_IMAGE in df,
        "found" if EXPECTED_BASE_IMAGE in df else "not found in Dockerfile",
    ))

    cmd_ok = any(form in df for form in _CMD_FORMS)
    results.append(_check(
        'Dockerfile CMD: python3 main.py',
        cmd_ok,
        "found" if cmd_ok else "not found in Dockerfile",
    ))

    results.append(_check(
        ".dockerignore exists",
        bool(di),
        ".dockerignore" if di else "MISSING",
    ))

    models_excluded = any(
        line.strip() in ("models/", "models", "models/**")
        for line in di.splitlines()
        if not line.strip().startswith("#")
    )
    results.append(_check(
        ".dockerignore does NOT exclude models/",
        not models_excluded,
        "OK — models/ will be included in build context" if not models_excluded else "PROBLEM — models/ is excluded",
    ))

    pt_excluded = any(
        line.strip() in ("*.pt", "**/*.pt", "models/**/*.pt")
        for line in di.splitlines()
        if not line.strip().startswith("#")
    )
    results.append(_check(
        ".dockerignore does NOT exclude *.pt",
        not pt_excluded,
        "OK — .pt files reachable for COPY" if not pt_excluded else "PROBLEM — *.pt excluded",
    ))

    results.append(_check(
        f"Local model weights exist: {MODEL_PATH.relative_to(REPO_ROOT)}",
        MODEL_PATH.exists(),
        f"{MODEL_PATH.stat().st_size // 1024 // 1024} MB" if MODEL_PATH.exists() else "MISSING — run training first",
    ))

    results.append(_check(
        "Model weights are gitignored",
        _gitignore_ignores_model(),
        "OK — will not be committed" if _gitignore_ignores_model() else "WARNING — may be committed",
    ))

    req = REPO_ROOT / "requirements.txt"
    results.append(_check(
        "requirements.txt exists",
        req.exists(),
    ))

    print("=" * 40)
    passed = sum(results)
    total = len(results)
    print(f"{'PASS' if passed == total else 'FAIL'}: {passed}/{total} checks passed.")

    if passed < total:
        print("\nFix the FAIL items above before running docker build.")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
