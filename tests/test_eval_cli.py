"""CLI contract for scripts/eval_recommender.py argument validation.

Runs the script as a subprocess so argparse behavior (usage + exit code 2)
is exercised end to end. Never invokes a valid --limit — that path loads
the embedding model.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "eval_recommender.py"
FIXTURE = ROOT / "tests" / "fixtures" / "borrow-sequences.json"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_limit_zero_is_rejected_with_user_facing_message():
    result = run_cli("--sequences", str(FIXTURE), "--limit", "0")
    assert result.returncode == 2
    assert "--limit must be between 1 and 50" in result.stderr
    assert "hit_at" not in result.stderr


def test_limit_above_list_size_is_rejected_with_user_facing_message():
    result = run_cli("--sequences", str(FIXTURE), "--limit", "51")
    assert result.returncode == 2
    assert "--limit must be between 1 and 50" in result.stderr
    assert "hit_at" not in result.stderr


def test_missing_sequences_file_still_errors_cleanly():
    result = run_cli("--sequences", "does-not-exist.json", "--limit", "10")
    assert result.returncode == 2
    assert "cannot read" in result.stderr
