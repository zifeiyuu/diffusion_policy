#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
policy_python="${ROBODIFF_PYTHON:-/home/zxiao93/anaconda3/envs/robodiff/bin/python}"
result_dir=/home/zxiao93/Documents/LaDiWM/results/official_dp_square_20260922
export DP_REPO_ROOT="$repo_dir"
export LD_LIBRARY_PATH="${HOME}/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
mkdir -p "$result_dir"
"$policy_python" scripts/audit_official_dp_resets.py > "$result_dir/official_reset_audit.log" 2>&1
for run in 0 1 2; do
  for split in valid random_saved; do
    echo "START run=$run split=$split $(date -Iseconds)"
    "$policy_python" scripts/eval_official_dp_square.py --run "$run" --split "$split" --batch 8 > "$result_dir/official_eval_${run}_${split}.log" 2>&1
    echo "DONE run=$run split=$split $(date -Iseconds)"
  done
done
"$policy_python" scripts/report_official_dp_square.py
