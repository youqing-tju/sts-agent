import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_verify_env_runs():
    r = subprocess.run(
        [sys.executable, "verify_env.py", "12345"],
        capture_output=True, text=True, timeout=300, cwd=ROOT,
    )
    assert r.returncode == 0, f"stderr: {r.stderr[-2000:]}"
    assert "outcome=" in r.stdout
