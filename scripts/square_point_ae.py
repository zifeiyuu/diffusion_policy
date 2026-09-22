"""Frozen encoder compatible with LaDiWM/K-Steer Autoencoder_V2, latent128.

Absolute pixel XY, NOT normalized coordinates or displacement. Architecture
matches AutoEncoder(in_channels=2,latent_dim=2,input_size=16,target_size=8).
"""
import hashlib
import os
from pathlib import Path
import torch
from torch import nn

AE_PATH=Path(os.environ.get('DP_AE_PATH',str(Path(__file__).resolve().parents[1]/'assets/point_ae/square_point_ae.pt')))
EXPECTED_SHA='c43c14af5d9881621d2306a4f0d3b3120d9b9027d7301f7ccdb2b3fbc072300f'


def ae_sha():
    actual=hashlib.sha256(AE_PATH.read_bytes()).hexdigest()
    if actual!=EXPECTED_SHA:raise ValueError('PointAE identity differs from selected LaDiWM checkpoint')
    return actual


class FrozenPointAE(nn.Module):
    def __init__(self):
        super().__init__();ae_sha()
        self.encoder=nn.Sequential(nn.Conv2d(2,8,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(8,2,1))
        state=torch.load(AE_PATH,map_location='cpu')
        state={k[len('encoder.'):]:v for k,v in state.items() if k.startswith('encoder.')}
        self.load_state_dict(state,strict=True)
        self.requires_grad_(False);self.eval()

    @torch.no_grad()
    def forward(self,points):
        assert points.shape[-2:]==(256,2)
        return self.encoder(points.reshape(-1,16,16,2).permute(0,3,1,2)).flatten(1)
