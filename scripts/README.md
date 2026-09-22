# LaDiWM-aligned Square Diffusion Policy

Run or resume the six-job queue:

```bash
bash scripts/run_square_aligned.sh
```

The queue uses the existing `robodiff` training environment and `kguide` simulator environment. It trains image and causal point-flow policies from scratch for 3050 epochs each, seeds42/43/44, sequentially on one GPU. Each final EMA checkpoint is evaluated on LaDiWM's exact20 validation initial states and saved50 random initial states before advancing. It updates the user-selected block in `LaDiWM/experiment_logs/summary/README.md` and registers evidence in Others. A nonblocking file lock prevents duplicate queues. No external logging service is used.

Artifacts: `/home/zxiao93/Documents/LaDiWM/results/dp_square_3050_20260920/`. Inspect `queue_status.json`, `queue.log`, per-job `.log`, and per-run `progress.json` / `epochs.jsonl`. Epoch checkpoints are atomic and saved at epoch0 and every10 epochs, plus the final epoch. Restart the queue to resume the last saved epoch. The final EMA and its SHA256 are retained; redundant optimizer snapshots are removed only after both evaluations complete. Failed subprocesses stop the queue and record failure instead of skipping a seed.

The model, diffusion loss, sampling, and EMA implementations are imported directly from this official DP checkout. The adapter changes the data split to HDF5 masks180/20, fits normalizers on train180 only, and evaluates final EMA rather than selecting by rollout SR. It retains the upstream horizon16/pad1/7 training windows, two observed frames, execute8, DDPM100, and UNet512/1024/2048. The training loop is local, not the unmodified upstream Hydra workspace. Validation uses evaluation-mode crops and independent RNG. Data shuffling uses a per-epoch generator for reproducible epoch resume. There are no periodic simulator rollouts during training.

Image input: two RGB84 views, random crop76 / center crop, independent ResNet18+SpatialSoftmax encoders, random weights. Point-flow replaces their combined128D output with an MLP1024→256→128. Coordinates are the same256 fixed surface projections as LaDiWM, including occluded points, plus backward displacement; the older observation token has zero displacement so both variants access only two frames. PointAE is not loaded. EE XYZ3/quaternion4/measured gripper qpos2 and all action diffusion settings are shared. This does not make privileged geometry equivalent to RGB; report input and encoder differences explicitly.

Verification:

```bash
/home/zxiao93/anaconda3/envs/robodiff/bin/python scripts/audit_square_aligned.py
```

The audit checks every training action window against upstream indexing, disjoint normalizer rows, causal t0 padding, live two-camera image/point/robot parity with HDF5, and bank state restoration. Separate smoke trainers/evaluators are stored under `smoke_*`; they never enter reported final SRs. Formal evaluation verifies all initial states and logs controller-bound clipping caused by DDPM terminal floating-point roundoff.

Environment repairs: installed `robomimic==0.2.0` and pinned `huggingface-hub==0.25.2` to restore `cached_download` compatibility with existing `diffusers==0.11.1`. Robosuite1.4.1 runs in `kguide`; this does not install the original mujoco-py/robosuite stack into `robodiff`.

Reporting includes train/evaluator wall cost, peak GPU allocation, inference mean/P95, SR, and task makespan on both sets. Success makespan is steps to first success (seconds=steps/20), summarized by mean/median/P95; failure durations are censored and listed separately. Execution wall includes policy/simulator/observation and in-loop video time, excluding reset/startup/final writes. Three-seed means and sample SDs are produced after all three seeds finish. No-success makespan is N/A.
