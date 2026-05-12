#!/usr/bin/env bash
# Launch the model-size sweep in a single detached tmux session.
#
# Usage:
#   scripts/launch_sweep.sh [extra args forwarded to python -m errorboard.sweep]
#
# The whole sweep runs in one tmux session (errorboard-sweep), sequentially
# training each cell. Each cell's outputs land in runs/sweep-L{n}-E{n}/.

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

session_name="errorboard-sweep"
if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "Error: tmux session '$session_name' already exists." >&2
    echo "  kill:   tmux kill-session -t '$session_name'" >&2
    echo "  attach: tmux attach -t '$session_name'" >&2
    exit 1
fi

mkdir -p "$repo_root/runs"
log_path="$repo_root/runs/sweep.log"

cmd="cd '$repo_root/code' \
    && source '$repo_root/.venv/bin/activate' \
    && python -u -m errorboard.sweep --runs-dir '$repo_root/runs' $* \
    2>&1 | tee -a '$log_path'"

tmux new-session -d -s "$session_name" "$cmd"

echo "Launched tmux session: $session_name"
echo "  log:     $log_path"
echo "  tail:    tail -f '$log_path'"
echo "  attach:  tmux attach -t '$session_name'"
echo "  status:  (cd $repo_root/code && python -m errorboard.status)"
echo "  summary: (cd $repo_root/code && python -m errorboard.sweep --summary)"
