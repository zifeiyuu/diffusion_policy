#!/usr/bin/env bash
set -euo pipefail
cd /home/zxiao93/Documents/diffusion_policy
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
exec /home/zxiao93/anaconda3/envs/robodiff/bin/python -u scripts/queue_square_aligned.py
