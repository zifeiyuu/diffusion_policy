## Square Image Transformer DP baseline:50 epochs,batch128

Fresh seed42 training with the upstream Transformer Hybrid Image policy:8 layers,4 heads,embedding256;8,997,642 action-network parameters plus22,394,176 image-encoder parameters (31,391,818 trainable total). Two84x84 RGB views and robot9 over two observation frames; image encoders train jointly from random initialization. Absolute10D policy actions,horizon10,execute8,DDPM100,EMA. Preserve Transformer AdamW defaults (LR1e-4,betas0.9/0.95,weight decay1e-3 Transformer /1e-6 image encoder),warmup1000,and cosine schedule over3050 epochs; stop at50 completed epochs. Only training batch changes to128; validation batch remains64. Train/validation masks180/20; evaluate exact Eval20 and saved Random50. No initial rollout or best-SR selection; evaluate final epoch49 EMA only. Training50 epochs at batch128 uses about half the updates of50 epochs at batch64.

Training/policy run in robodiff; simulation uses dp_eval141 (robosuite1.4.1) with full saved XML/state. Shared data and absolute-action cache are reused read-only from the prior study; action labels were generated with the upstream converter in robodiff1.2. Outputs and this report are stored in diffusion_policy, not LaDiWM. Checkpoint: `data/outputs/square_transformer_b128_e50_seed42/checkpoints/last_epoch_0049.ckpt` (model,EMA,optimizer); inference weights: `last/policy.pt` under the same output directory.

Trainer wall includes setup,offline validation,diagnostic sampling and checkpointing; no simulator rollout occurs during training. Final evaluator wall is separate and includes initialization,inference,simulation,rendering and saving. GPU allocation,batched and single-env inference timings,and success-only task makespan are recorded in result JSON. Failed episodes are censored at400 controls/20Hz. Reused AE/Stage1 costs are not applicable to this image model; no monetary/energy measurements.

Status: `{'state': 'waiting', 'job': 'Waiting for existing GPU compute jobs to finish', 'updated': 1790208033.068606}`.

| Train epochs | Batch | Eval20 SR | Random50 SR | Train wall min | Eval20 / Random50 wall min |
|---:|---:|---:|---:|---:|---:|
| 50 | 128 | Pending | Pending | Pending | Pending / Pending |
