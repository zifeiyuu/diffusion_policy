# Square DP on Skynet

`run_skynet.sh` follows the partition/account/A40/64GB/48-hour example in `others`, but launches this repository's Square DP trainer. Cluster access and current partition permissions have not been tested. No jobs were submitted remotely. Training stays stopped on Sirius.

## Models and budget

Both encoders initialize randomly and learn jointly from the action denoising loss. Image uses two ResNet18 + SpatialSoftmax encoders; point-flow uses MLP1024→256→128 over ordered projected XY and causal displacement. Neither loads PointAE nor ImageNet weights. Both use Square PH200, HDF5 train180/valid20,7D delta OSC,2 observation frames,predict16/execute8,DDPM100.

The official checked-in hybrid configuration sets3050 epochs; the paper describes3000 for image benchmarks. This is a benchmark budget, not a minimum convergence requirement. The stopped Sirius job measured about66–70 seconds per full epoch including validation on RTX4080SUPER: about56–60 hours for3050,1.8–2 hours for100,5.5–6 hours for300 **per image seed**, excluding online evaluation. A40 timing must be measured; these estimates do not promise A40 speed or extrapolate point-flow training. A100-epoch pilot is reasonable before committing six long runs.

Budgets are separate experiments with separate cosine schedules. Do not resume a3050-epoch checkpoint with `--epochs100`; the trainer rejects changing the recorded budget. Use a new output root. The historical3050-only local report script is intended for that experiment; do not use it to label shorter pilot results. Raw evaluation JSON contains SR, train/inference costs and task makespan for all budgets; integrate pilot results into LaDiWM under a new experiment entry.

## Push code to your own remote

Current `origin` is the upstream `https://github.com/real-stanford/diffusion_policy.git`. The personal repository is `https://github.com/zifeiyuu/diffusion_policy.git`, branch `main`. The local `personal` remote fetches over HTTPS and pushes over SSH. For subsequent code updates:

```bash
cd /home/zxiao93/Documents/diffusion_policy
git add .gitignore run_skynet.sh scripts/
git diff --cached --stat
git commit -m "Update Square DP tooling"
git push personal main
```

`others/` contains reference scripts and is not required at runtime; add it separately only if you want to publish it. The new ignore patterns exclude HDF5, weights, arrays, videos and Slurm logs. Existing tracked files are unaffected. Never add the dataset, cache, checkpoints, or Conda directories to Git.

On Skynet:

```bash
mkdir -p /nethome/zxiao93/code
cd /nethome/zxiao93/code
git clone git@github.com:zifeiyuu/diffusion_policy.git
cd diffusion_policy
mkdir -p skynet_logs  # Must exist BEFORE sbatch opens its log files.
```

## Files transferred outside Git

A ready-to-upload local bundle is available at `/home/zxiao93/Documents/diffusion_policy/skynet_upload/square/` (254 real files, about2.66GiB). Upload its contents to `/coc/flash2/zxiao93/datasets/robomimic/square/`. It includes `SHA256SUMS`; run `sha256sum -c SHA256SUMS` in the remote square directory after upload. This local bundle is ignored by Git.

| Artifact | Sirius location, relative to Square PH root | Measured size | Required |
|---|---|---:|---|
| RGB demonstrations | `source/image_v141.hdf5` |2.5GiB|Yes, shared cache preparation reads both variants|
| Robot states/actions/reset metadata | `source/low_dim_v141.hdf5` |51MiB|Yes|
| Projected surface points | `gt_geometry/episodes/` |188MiB|Yes|
| Geometry identity | `gt_geometry/anchors_body.npy`, `gt_geometry/manifest.json` |Small|Yes|
| Saved Random50 resets |50 `initial_*.npz` files listed below|400KiB total disk usage|Yes, cache records their identities|
| Existing resume checkpoint | `LaDiWM/results/dp_square_3050_20260920/image_seed42/latest.pt` |4.2GiB|Optional; epoch0 only, budget3050|
| Generated shared cache | same run root `/cache/` |1.4GiB|Optional; rebuilt on Skynet|

No PNG image tree, PointAE weights, Koopman weights, pre-rendered videos, or old experiment outputs are required for training. Reserve space for atomic checkpoint writes: each active image job can temporarily require about9GiB for old/new optimizer snapshots, plus final EMA(~1GiB), cache and dataset. Multiple independent jobs multiply checkpoint storage.

Run these on Sirius; replace `SKYNET_HOST` with your actual SSH alias/hostname:

```bash
ssh zxiao93@SKYNET_HOST 'mkdir -p /coc/flash2/zxiao93/datasets/robomimic/square/ph/source /coc/flash2/zxiao93/datasets/robomimic/square/ph/gt_geometry /coc/flash2/zxiao93/datasets/robomimic/square/random50'
rsync -avP /home/zxiao93/Documents/Datasets/robomimic/square/ph/source/{image_v141,low_dim_v141}.hdf5 zxiao93@SKYNET_HOST:/coc/flash2/zxiao93/datasets/robomimic/square/ph/source/
rsync -avP /home/zxiao93/Documents/Datasets/robomimic/square/ph/gt_geometry/{episodes,anchors_body.npy,manifest.json} zxiao93@SKYNET_HOST:/coc/flash2/zxiao93/datasets/robomimic/square/ph/gt_geometry/
rsync -avP /home/zxiao93/Documents/LaDiWM/results/kubm_square_all200/eval/20260916_230541/initial_*.npz zxiao93@SKYNET_HOST:/coc/flash2/zxiao93/datasets/robomimic/square/random50/
```

Do not upload the local Conda directory as ordinary files. Recreate or properly relocate the environment. The tested Sirius trainer uses the repository's torch1.12.1 environment, `robomimic==0.2.0`, `diffusers==0.11.1`, and `huggingface-hub==0.25.2`; the last pin is needed for `cached_download`. `conda_environment.yaml` includes additional simulator packages and is not a guarantee of a clean installation on a new cluster. Run the smoke job before submitting a long budget. Training itself does not import robosuite or need the LaDiWM repository.

## Submit and resume

Default Python: `/nethome/zxiao93/anaconda3/envs/robodiff/bin/python`. Override with `DP_TRAIN_PYTHON`. Other overrides: `DP_REPO_ROOT`, `DP_DATA_ROOT`, `DP_BANK_ROOT`, `DP_OUTPUT_ROOT`. Export them before sbatch. Default output root is `/coc/flash2/zxiao93/diffusion_policy/square_${EPOCHS}epochs`.

```bash
# Verify the environment and two training updates; no meaningful SR is implied.
sbatch --time=00:20:00 run_skynet.sh verify image 42 100
sbatch --time=00:20:00 run_skynet.sh verify point_flow 42 100

# Pilot: two independent jobs, if cluster quota permits.
sbatch run_skynet.sh train image 42 100
sbatch run_skynet.sh train point_flow 42 100

# Optional: all six original-budget runs. Each job requests one GPU.
for modality in image point_flow; do
  for seed in 42 43 44; do
    sbatch run_skynet.sh train "$modality" "$seed" 3050
  done
done
```

The trainer finishes the current epoch, atomically saves model/EMA/optimizer/scheduler/RNG states, then exits75 fifteen minutes before the Slurm EndTime. The launcher submits an afterok continuation and exits successfully. Actual overridden walltime is carried to the continuation; its partition/GPU/account still come from this script's directives. If using different resources, edit those directives consistently. Default auto-continuation limit:100 jobs. Set `DP_AUTO_CONTINUE=0` to disable submission; rerun the identical train command manually to resume. `scancel` or abrupt crashes can lose progress since the latest saved epoch. Locks prevent concurrent jobs writing the same run; a separate lock serializes first-time cache preparation.

Inspect `skynet_logs/` and `$DP_OUTPUT_ROOT/{image,point_flow}_seed*/{progress.json,epochs.jsonl}`. Full trainer wall, epoch compute and peak GPU are recorded in `complete.json`. Exited job queue waiting time is excluded from recorded training wall.

## Evaluation and reporting

Training does **not** automatically submit evaluation jobs. After training finishes:

```bash
sbatch run_skynet.sh eval image 42 100
sbatch run_skynet.sh eval point_flow 42 100
```

Evaluation additionally requires the LaDiWM checkout containing `ladiwm/kubm/geometry_source.py` and a working robosuite1.4.1 simulator Python. Defaults: `DP_LADIWM_ROOT=/nethome/zxiao93/code/LaDiWM`, `DP_SIM_PYTHON=/nethome/zxiao93/anaconda3/envs/kguide/bin/python`. Configure these explicitly if different. `kguide` is only an environment name: the simulator version/data compatibility matters. Evaluation does not load PointAE/KUBM weights. Its jobs do not yet auto-resume partial episodes; a failed split restarts that split, while a completed split is skipped.

Alternatively download run folders and the unchanged shared cache manifest to Sirius, set `DP_OUTPUT_ROOT` to the downloaded budget root, and execute `scripts/eval_square_aligned.py --run ... --split valid` and `--split random_saved` using Sirius robodiff/kguide. Preserve the cache `manifest.json` byte-for-byte: its SHA256 is tied to checkpoints. Restore data and bank paths using the documented environment variables.

Both splits write `results.json`, initial states, trajectories and first-three-episode videos. Records include success rate, full evaluator wall, inference mean/P95 and summed inference time, peak GPU, and task makespan (success steps/20Hz and execution wall; failed episodes reported separately). Sync those artifacts and training `complete.json` back into LaDiWM for the combined report. Shorter-budget experiments must receive their own report/IDs rather than overwriting the3050 baseline.
