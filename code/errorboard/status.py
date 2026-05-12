"""Print the status of all runs in runs/.

Designed for re-orienting in a fresh Claude / shell session:
    python -m errorboard.status [--runs-dir runs]

Shows STATUS, last iter, last natural-distribution loss, and whether the
associated tmux session is still alive.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _tmux_session_alive(name: str) -> bool:
    try:
        r = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True, timeout=2,
        )
        return r.returncode == 0
    except Exception:
        return False


def _last_metrics_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    last = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    if last is None:
        return None
    try:
        return json.loads(last)
    except Exception:
        return None


def report(runs_dir: Path) -> None:
    if not runs_dir.exists():
        print(f"No runs/ directory at {runs_dir}")
        return

    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    if not run_dirs:
        print(f"No runs under {runs_dir}/")
        return

    print(f"{'run':<30} {'status':<10} {'tmux':<5} {'iter':>7} {'nat_loss':>10}")
    print("-" * 70)
    for d in run_dirs:
        status_file = d / "STATUS"
        status = status_file.read_text().strip() if status_file.exists() else "?"
        tmux_alive = "yes" if _tmux_session_alive(f"errorboard-{d.name}") else "no"
        last = _last_metrics_row(d / "metrics.jsonl")
        if last:
            it = f"{last.get('iter', '?'):>7}"
            nl = last.get("natural_loss")
            nl_s = f"{nl:>10.4f}" if isinstance(nl, (int, float)) else f"{'?':>10}"
        else:
            it, nl_s = f"{'-':>7}", f"{'-':>10}"
        print(f"{d.name:<30} {status:<10} {tmux_alive:<5} {it} {nl_s}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs", type=Path)
    args = p.parse_args()
    report(args.runs_dir)


if __name__ == "__main__":
    main()
