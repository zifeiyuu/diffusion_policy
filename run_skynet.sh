#!/usr/bin/env bash
#SBATCH --job-name=square_dp
#SBATCH --partition=ravichandar-lab
#SBATCH --account=ravichandar-lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a40:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --qos=short
#SBATCH --output=skynet_logs/%x.%j.out
#SBATCH --error=skynet_logs/%x.%j.err

set -euo pipefail
# Run sbatch from the checkout, not from the dataset directory.
REPO=${DP_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}}
cd "$REPO"
export DP_REPO_ROOT="$REPO"
PYTHON=${DP_TRAIN_PYTHON:-/nethome/zxiao93/anaconda3/envs/robodiff/bin/python}
MODE=${1:-train}
MODALITY=${2:-image}
SEED=${3:-42}
EPOCHS=${4:-3050}
CONTINUATIONS=${5:-0}
case "$MODE" in prepare|verify|train|eval) ;; *) echo 'Mode must be prepare, verify, train, or eval' >&2; exit 2;; esac
case "$MODALITY" in image|point_flow|point_ae) ;; *) echo 'Modality must be image, point_flow, or point_ae' >&2; exit 2;; esac
[[ "$SEED" =~ ^[0-9]+$ && "$EPOCHS" =~ ^[1-9][0-9]*$ && "$CONTINUATIONS" =~ ^[0-9]+$ ]] || exit 2
export DP_DATA_ROOT=${DP_DATA_ROOT:-/coc/flash2/zxiao93/datasets/robomimic/square/ph}
export DP_BANK_ROOT=${DP_BANK_ROOT:-/coc/flash2/zxiao93/datasets/robomimic/square/random50}
export DP_OUTPUT_ROOT=${DP_OUTPUT_ROOT:-/coc/flash2/zxiao93/diffusion_policy/square_${EPOCHS}epochs}
export DP_LADIWM_ROOT=${DP_LADIWM_ROOT:-/nethome/zxiao93/code/LaDiWM}
export DP_SIM_PYTHON=${DP_SIM_PYTHON:-/nethome/zxiao93/anaconda3/envs/kguide/bin/python}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl WANDB_MODE=disabled
mkdir -p "$DP_OUTPUT_ROOT" skynet_logs
[[ -x "$PYTHON" ]] || { echo "Missing training Python: $PYTHON" >&2; exit 2; }
for f in source/image_v141.hdf5 source/low_dim_v141.hdf5 gt_geometry/anchors_body.npy gt_geometry/manifest.json; do
    [[ -f "$DP_DATA_ROOT/$f" ]] || { echo "Missing data: $DP_DATA_ROOT/$f" >&2; exit 2; }
done
for i in $(seq 0 49); do
    printf -v name 'initial_%03d.npz' "$i"
    [[ -f "$DP_BANK_ROOT/$name" ]] || { echo "Missing reset bank: $name" >&2; exit 2; }
done
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print("GPU:",torch.cuda.get_device_name()); from diffusers import DDPMScheduler; import robomimic'
# Different seeds may run concurrently; identical jobs must never write one checkpoint.
exec 9>"$DP_OUTPUT_ROOT/${MODE}_${MODALITY}_${SEED}.lock"
flock -n 9 || { echo 'This job is already running' >&2; exit 2; }
# Keep fifteen minutes for finishing the current epoch and writing the checkpoint.
DEADLINE_ARGS=()
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    JOB_INFO=$(scontrol show job "$SLURM_JOB_ID" -o)
    JOB_END=$("$PYTHON" -c 'import re,sys; print(re.search(r"(?:^| )EndTime=([^ ]+)",sys.stdin.read()).group(1))' <<< "$JOB_INFO")
    DEADLINE=$(( $(date -d "$JOB_END" +%s) - 900 ))
    DEADLINE_ARGS=(--deadline "$DEADLINE")
fi
RUN="$DP_OUTPUT_ROOT/${MODALITY}_seed${SEED}"
case "$MODE" in
  prepare)
    "$PYTHON" -c 'import sys;sys.path.insert(0,"scripts");from square_dp_data import prepare;print(prepare()["splits"])'
    ;;
  verify)
    CHECK="$DP_OUTPUT_ROOT/verification/${SLURM_JOB_ID:-manual}/${MODALITY}_seed${SEED}"
    "$PYTHON" scripts/train_square_aligned.py --modality "$MODALITY" --seed "$SEED" --epochs 1 --max-batches 2 --output "$CHECK"
    echo "Smoke training passed. Not a success-rate evaluation. Artifacts: $CHECK"
    ;;
  train)
    result=0
    "$PYTHON" -u scripts/train_square_aligned.py --modality "$MODALITY" --seed "$SEED" --epochs "$EPOCHS" "${DEADLINE_ARGS[@]}" || result=$?
    if [[ "$result" == 75 ]]; then
        if [[ -n ${SLURM_JOB_ID:-} && ${DP_AUTO_CONTINUE:-1} == 1 && "$CONTINUATIONS" -lt 100 ]]; then
            # Preserve actual allocation walltime even when it was overridden at submission.
            JOB_START=$("$PYTHON" -c 'import re,sys; print(re.search(r"(?:^| )StartTime=([^ ]+)",sys.stdin.read()).group(1))' <<< "$JOB_INFO")
            JOB_MINUTES=$(( ( $(date -d "$JOB_END" +%s) - $(date -d "$JOB_START" +%s) ) / 60 ))
            next_job=$(sbatch --parsable --export=ALL --time="$JOB_MINUTES" --dependency="afterok:$SLURM_JOB_ID" "$REPO/run_skynet.sh" train "$MODALITY" "$SEED" "$EPOCHS" "$((CONTINUATIONS+1))")
            echo "Checkpoint saved; continuation job: $next_job"
            exit 0
        fi
        echo "Checkpoint saved. Resume: sbatch run_skynet.sh train $MODALITY $SEED $EPOCHS"
    fi
    exit "$result"
    ;;
  eval)
    [[ -x "$DP_SIM_PYTHON" && -f "$DP_LADIWM_ROOT/ladiwm/kubm/geometry_source.py" ]] || {
        echo 'Evaluation additionally needs kguide/robosuite1.4.1 and LaDiWM; see scripts/SKYNET.md.' >&2; exit 2;
    }
    for split in valid random_saved; do
        "$PYTHON" -u scripts/eval_square_aligned.py --run "$RUN" --split "$split"
    done
    ;;
esac
