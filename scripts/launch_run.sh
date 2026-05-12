#!/usr/bin/env bash
# Launch an ErrorBoard training run in a detached tmux session.
#
# Usage:
#   scripts/launch_run.sh <run_name> [extra training args...]
#
# Sets up tmux session `errorboard-<run_name>`, runs training with stdout/stderr
# tee'd to runs/<run_name>/train.log, and creates the run directory if needed.
# Errors out on tmux session name collision (prevents accidental clobber).
#
# After launching:
#   tmux ls                                       # confirm session is up
#   tail -f runs/<run_name>/train.log             # watch live
#   python -m errorboard.status                   # all-runs status (from code/)
#   tmux attach -t errorboard-<run_name>          # interactive attach
#   tmux kill-session -t errorboard-<run_name>    # kill

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "usage: $0 <run_name> [training args...]" >&2
    exit 1
fi

run_name="$1"; shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

run_dir="$repo_root/runs/$run_name"
mkdir -p "$run_dir"

session_name="errorboard-$run_name"
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Error: tmux session '$session_name' already exists." >&2
    echo "  kill:   tmux kill-session -t '$session_name'" >&2
    echo "  attach: tmux attach -t '$session_name'" >&2
    exit 1
fi

# The training module is in code/errorboard/, so we cd to code/ and use python -u
# for unbuffered output (so the tee'd log updates live).
cmd="cd '$repo_root/code' \
    && source '$repo_root/.venv/bin/activate' \
    && python -u -m errorboard.training \
        --run-name '$run_name' \
        --runs-dir '$repo_root/runs' \
        $* \
    2>&1 | tee -a '$run_dir/train.log'"

tmux new-session -d -s "$session_name" "$cmd"

echo "Launched tmux session: $session_name"
echo "  run dir:    $run_dir"
echo "  attach:     tmux attach -t '$session_name'"
echo "  tail log:   tail -f '$run_dir/train.log'"
echo "  status:     cat '$run_dir/STATUS'"
echo "  all runs:   (cd $repo_root/code && python -m errorboard.status)"
