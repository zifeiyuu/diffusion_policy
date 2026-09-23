"""Only explicit LaDiWM split, frozen AE observation, and evaluation adapters."""
import copy, json, subprocess, sys, time
from pathlib import Path
import h5py, hydra, numpy as np, torch
from torch import nn
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval",eval,replace=True)
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from square_dp_data import ROBOT

class MaskDataset(RobomimicReplayImageDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with h5py.File(kwargs['dataset_path']) as f:
            valid={v.decode() for v in f['mask/valid'][:]}
            train={v.decode() for v in f['mask/train'][:]}
        assert len(train)==180 and len(valid)==20 and not train&valid
        self.train_mask=np.array([f'demo_{i}' in train for i in range(self.replay_buffer.n_episodes)])
        self.sampler=SequenceSampler(self.replay_buffer,self.horizon,pad_before=self.pad_before,pad_after=self.pad_after,episode_mask=self.train_mask,key_first_k={k:self.n_obs_steps for k in self.rgb_keys+self.lowdim_keys})

class AEConcat(nn.Module):
    def forward(self,obs):return torch.cat([obs['point_ae_qpos']]+[obs[k] for k in ROBOT],dim=-1)

def ae_policy(base):
    # Instantiate exactly the image policy; replace only its 128D visual encoder.
    p=hydra.utils.instantiate(base)
    assert p.obs_feature_dim==137
    p.obs_encoder=AEConcat()
    return p

class PairedRunner(BaseImageRunner):
    def __init__(self,output_dir,modality,down_dims=None,architecture="unet"):
        super().__init__(output_dir);self.modality=modality;self.index=1;self.down_dims=list(down_dims) if down_dims is not None else None;self.architecture=architecture
    def run(self,policy):
        epoch=self.index*50;self.index+=1
        folder=Path(self.output_dir)/'rollouts'/f'epoch_{epoch:04d}';folder.mkdir(parents=True,exist_ok=True)
        start=time.perf_counter()
        # Only EMA weights are copied for evaluation; training state remains upstream.
        state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        torch.save(dict(model=state,config=dict(modality=self.modality,seed=42,epochs=100,smoke=False,down_dims=self.down_dims,architecture=self.architecture),epoch=epoch),folder/'policy.pt');del state
        origin=Path(self.output_dir)/'train_started.json'
        if origin.exists():
            (folder/'training_time.json').write_text(json.dumps({'epochs_completed':epoch+1,'trainer_wall_seconds_before_rollout':time.time()-json.loads(origin.read_text())['unix_time']})+'\n')
        for split in ['valid','random_saved']:
            with (folder/f'{split}.log').open('a') as f:
                subprocess.run([sys.executable,'scripts/eval_square_upstream.py','--run',str(folder),'--split',split],stdout=f,stderr=subprocess.STDOUT,check=True)
        valid=json.loads((folder/'valid/results.json').read_text());test=json.loads((folder/'random_saved/results.json').read_text())
        (folder/'rollout_wall.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-start})+'\n')
        return {'test/mean_score':test['success_rate'],'validation/mean_score':valid['success_rate']}

def config(modality,down_dims=None,architecture="unet"):
    from hydra import initialize_config_dir, compose
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'diffusion_policy/config'),version_base=None):
        cfg=compose(config_name='train_diffusion_transformer_hybrid_workspace' if architecture=='transformer' else 'train_diffusion_unet_hybrid_workspace',overrides=['task=square_image_abs'])
    OmegaConf.set_struct(cfg,False)
    original=OmegaConf.to_container(cfg,resolve=True)
    cfg.training.skip_initial_rollout=True
    cfg.training.stop_after_epochs=100  # scheduler still sees original num_epochs=3050
    cfg.logging.mode='offline'
    cfg.task.dataset._target_='square_upstream_adapters.MaskDataset'
    cfg.task.dataset.dataset_path='/home/zxiao93/Documents/LaDiWM/results/dp_upstream_abs100_20260922/image_abs.hdf5'
    # Separate caches because the replay keys differ; values come from the same source.
    if modality=='point_ae':
        cfg.task.dataset.dataset_path='/home/zxiao93/Documents/LaDiWM/results/dp_upstream_abs100_20260922/point_ae_abs.hdf5'
    cfg.task.env_runner=OmegaConf.create(dict(_target_='square_upstream_adapters.PairedRunner',modality=modality))
    if architecture=='transformer':cfg.task.env_runner.architecture='transformer'
    if down_dims is not None:
        cfg.policy.down_dims=list(down_dims)
        cfg.task.env_runner.down_dims=list(down_dims)
    if modality=='point_ae':
        base=OmegaConf.to_container(cfg.policy,resolve=True)
        cfg.policy=OmegaConf.create({'_target_':'square_upstream_adapters.ae_policy','_recursive_':False,'base':base})
        obs={k:OmegaConf.to_container(cfg.task.shape_meta.obs[k],resolve=True) for k in ROBOT}
        obs['point_ae_qpos']={'shape':[128],'type':'low_dim'}
        cfg.task.shape_meta.obs=OmegaConf.create(obs)
        cfg.task.dataset.shape_meta=OmegaConf.to_container(cfg.task.shape_meta,resolve=True)
    return cfg,original
