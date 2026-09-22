# Frozen Square PointAE

`square_point_ae.pt` is the exact14KB checkpoint used by LaDiWM, copied from its `weights/` directory. Its recorded source is K-Steer `models/ae_square/final_model.pth`. SHA256: `c43c14af5d9881621d2306a4f0d3b3120d9b9027d7301f7ccdb2b3fbc072300f`.

This checkpoint is deliberately included in Git; no separate Skynet upload is necessary. Its supplied provenance says it was reused frozen, but does not identify the original AE training/validation split or training cost. Do not claim the AE never saw the20 validation demos.

The encoder receives256 ordered **absolute pixel XY coordinates**, reshaped `[B,16,16,2]→[B,2,16,16]`. It uses Conv2d2→8(kernel3,stride2,padding1),ReLU,Conv2d8→2(kernel1), producing `[B,2,8,8]`, flattened to128D. There is no coordinate normalization before the AE, no displacement input, no VAE sampling and no decoder use in DP. This matches LaDiWM `FrozenKUBM.encode_points`; DP does not use its Koopman lifting or forecast.

Our `point_ae` variant freezes this encoder, precomputes its128D outputs for offline training, normalizes those outputs using train180 only, and concatenates robot9 directly. It has no additional trainable observation encoder. Two latent history frames preserve causal observations. This is a separate pretrained absolute-point-latent experiment, not the same representation as the from-scratch coordinate-plus-displacement `point_flow` branch.
