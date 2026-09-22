"""LaDiWM-aligned data adapter; upstream DP model and action-window semantics."""
import hashlib
import os
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import (array_to_stats, get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat, get_image_range_normalizer)

LADIWM = Path(os.environ.get('DP_LADIWM_ROOT', '/home/zxiao93/Documents/LaDiWM'))
DATA = Path(os.environ.get('DP_DATA_ROOT', '/home/zxiao93/Documents/Datasets/robomimic/square/ph'))
OUT = Path(os.environ.get('DP_OUTPUT_ROOT', str(LADIWM / 'results/dp_square_3050_20260920')))
REPORT = Path(os.environ.get('DP_REPORT_ROOT', str(LADIWM / 'experiment_logs/2026-09-20/dp_square_3050')))
BANK = Path(os.environ.get('DP_BANK_ROOT', str(LADIWM / 'results/kubm_square_all200/eval/20260916_230541')))
ROBOT = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
RGB = ['agentview_image', 'robot0_eye_in_hand_image']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n'); tmp.replace(path)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'cache_prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare()


def _prepare():
    cache = OUT / 'cache'; cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / 'manifest.json'
    if manifest.exists():
        return json.loads(manifest.read_text())
    with h5py.File(DATA/'source/image_v141.hdf5') as f, h5py.File(DATA/'source/low_dim_v141.hdf5') as low:
        splits = {k: [n.decode() for n in f['mask'][k][:]] for k in ['train', 'valid']}
        for k, names in splits.items():
            assert names == [n.decode() for n in low['mask'][k][:]]
        assert len(splits['train']) == 180 and len(splits['valid']) == 20
        assert not set(splits['train']) & set(splits['valid'])
        names = sorted(f['data'], key=lambda n: int(n.split('_')[1]))
        lengths = [len(f['data'][n]['actions']) for n in names]
        ends = np.cumsum(lengths); total = int(ends[-1])
        shapes = {k: (total, *f['data/demo_0/obs'][k].shape[1:]) for k in ROBOT+RGB}
        shapes.update(action=(total,7), point_flow=(total,1024))
        arrays = {k: np.lib.format.open_memmap(cache/f'{k}.npy', mode='w+',
                  dtype='u1' if k in RGB else 'f4', shape=s) for k,s in shapes.items()}
        offset = 0
        for name, length in zip(names, lengths):
            d = f['data'][name]; ld = low['data'][name]
            np.testing.assert_array_equal(d['actions'][:], ld['actions'][:])
            np.testing.assert_array_equal(d['states'][:], ld['states'][:])
            with h5py.File(DATA/f'gt_geometry/episodes/{name}.hdf5') as g:
                np.testing.assert_array_equal(g['states'][:], ld['states'][:])
                np.testing.assert_array_equal(g['actions'][:], ld['actions'][:])
                p = g['points_xy'][:].astype('f4') / np.array([640,480], dtype='f4')
                # Backward difference only. Never use the stored forward flow_xy[t].
                dp = np.concatenate([np.zeros_like(p[:1]), np.diff(p,axis=0)])
                feature = np.concatenate([p.reshape(length,512),dp.reshape(length,512)],axis=-1)
            sl = slice(offset,offset+length)
            arrays['point_flow'][sl] = feature
            arrays['action'][sl] = d['actions'][:]
            for k in ROBOT+RGB:
                arrays[k][sl] = d['obs'][k][:]
            offset += length
        for a in arrays.values(): a.flush()
    info = dict(names=names, lengths=lengths, ends=ends.tolist(), splits=splits,
        data_sha256={k:digest(DATA/f'source/{k}_v141.hdf5') for k in ['image','low_dim']},
        geometry_manifest_sha256=digest(DATA/'gt_geometry/manifest.json'),
        anchors_sha256=digest(DATA/'gt_geometry/anchors_body.npy'),
        random_bank={f'initial_{i:03d}.npz':digest(BANK/f'initial_{i:03d}.npz') for i in range(50)},
        feature='256 ordered XY/[640,480] + causal (XY[t]-XY[t-1])/[640,480]; zero displacement at t0',
        normalization='180 training demonstrations only', source_dp_commit='5ba07ac')
    write_json(manifest,info)
    return info


def prepare_ae(info):
    from square_point_ae import FrozenPointAE, ae_sha
    cache=OUT/'cache';path=cache/'point_ae.npy';meta=cache/'point_ae_manifest.json'
    expected=dict(ae_sha256=ae_sha(),base_manifest_sha256=digest(cache/'manifest.json'),
                  input='absolute pixel XY reshaped16x16 in existing point order; no flow difference',
                  frozen=True,latent_dim=128,ae_pretraining_split='Not documented by supplied weight provenance')
    with (OUT/'ae_prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if meta.exists():
            assert json.loads(meta.read_text())==expected
            assert path.exists()
            return expected
        model=FrozenPointAE();tmp=cache/'point_ae.tmp.npy'
        result=np.lib.format.open_memmap(tmp,mode='w+',dtype='f4',shape=(info['ends'][-1],128))
        offset=0
        for name,length in zip(info['names'],info['lengths']):
            with h5py.File(DATA/f'gt_geometry/episodes/{name}.hdf5') as f:
                points=torch.from_numpy(f['points_xy'][:].astype('f4'))
            result[offset:offset+length]=model(points).numpy();offset+=length
        result.flush();del result;tmp.replace(path);write_json(meta,expected)
        return expected


class SquareDataset(Dataset):
    def __init__(self, modality, split, manifest=None):
        self.info = manifest or prepare(); self.modality=modality
        if modality=='point_ae': self.ae_info=prepare_ae(self.info)
        self.keys = ROBOT + (RGB if modality == 'image' else [modality])
        self.arrays = {k:np.load(OUT/'cache'/f'{k}.npy',mmap_mode='r') for k in self.keys+['action']}
        self.samples=[]; self.rows=[]; start=0
        for name, length in zip(self.info['names'],self.info['lengths']):
            if name in self.info['splits'][split]:
                self.rows.append(np.arange(start,start+length))
                # Exactly upstream horizon16 / pad_before1 / pad_after7.
                self.samples.extend((start,length,t) for t in range(length-7))
            start += length
        self.rows=np.concatenate(self.rows)

    def __len__(self): return len(self.samples)

    def __getitem__(self,index):
        start,length,t=self.samples[index]
        oi=start+np.maximum([t-1,t],0)
        ai=start+np.clip(np.arange(t-1,t+15),0,length-1)
        obs={}
        for k in self.keys:
            x=np.array(self.arrays[k][oi],dtype='f4')
            if k in RGB: x=np.moveaxis(x,-1,1)/255.
            if k == 'point_flow': x[0,512:] = 0.  # Same two-frame temporal access as RGB.
            obs[k]=torch.from_numpy(x)
        return dict(obs=obs,action=torch.from_numpy(np.array(self.arrays['action'][ai],dtype='f4')))

    def normalizer(self):
        n=LinearNormalizer()
        for k in ROBOT+['action'] + ([self.modality] if self.modality in ['point_flow','point_ae'] else []):
            stat=array_to_stats(np.array(self.arrays[k][self.rows]))
            n[k]=(get_identity_normalizer_from_stat(stat) if k in ['action','robot0_eef_quat']
                  else get_range_normalizer_from_stat(stat))
        if self.modality=='image':
            for k in RGB: n[k]=get_image_range_normalizer()
        return n


class PointFlowEncoder(nn.Module):
    """Replace the two 64D RGB embeddings with a 128D causal point-flow MLP."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Linear(256,128))
    def forward(self,obs):
        return torch.cat([self.net(obs['point_flow'])]+[obs[k] for k in ROBOT],dim=-1)


class CachedPointAEEncoder(nn.Module):
    """No trainable feature encoder: concatenate cached latent128 and robot9."""
    def forward(self,obs):
        return torch.cat([obs['point_ae']]+[obs[k] for k in ROBOT],dim=-1)


def make_policy(modality):
    shape_meta=dict(action=dict(shape=[7]),obs={
        **{k:dict(shape=[3,84,84],type='rgb') for k in RGB},
        **{k:dict(shape=[d],type='low_dim') for k,d in zip(ROBOT,[3,4,2])}})
    p=DiffusionUnetHybridImagePolicy(shape_meta=shape_meta,
        noise_scheduler=DDPMScheduler(num_train_timesteps=100,beta_start=.0001,beta_end=.02,
            beta_schedule='squaredcos_cap_v2',variance_type='fixed_small',clip_sample=True,prediction_type='epsilon'),
        horizon=16,n_action_steps=8,n_obs_steps=2,num_inference_steps=100,
        obs_as_global_cond=True,crop_shape=[76,76],diffusion_step_embed_dim=128,
        down_dims=[512,1024,2048],kernel_size=5,n_groups=8,cond_predict_scale=True,
        obs_encoder_group_norm=True,eval_fixed_crop=True)
    if modality=='point_flow':
        assert p.obs_feature_dim == 137, p.obs_feature_dim
        p.obs_encoder=PointFlowEncoder()
    elif modality=='point_ae':
        assert p.obs_feature_dim==137
        p.obs_encoder=CachedPointAEEncoder()
    print("Selected observation encoder parameters:",sum(x.numel() for x in p.obs_encoder.parameters()))
    return p
