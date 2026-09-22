"""Evaluate official Square image CNN checkpoints on paired LaDiWM reset sets."""
import argparse
from collections import deque
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import types
from multiprocessing.connection import Listener

ROOT=Path('/home/zxiao93/Documents/LaDiWM')
DP=Path(os.environ.get('DP_REPO_ROOT','/home/zxiao93/Documents/diffusion_policy'))
sys.path.insert(0,str(DP));sys.path.insert(0,str(DP/'scripts'))
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['LD_LIBRARY_PATH']=os.pathsep.join([str(Path.home()/'.mujoco/mujoco210/bin'),'/usr/lib/x86_64-linux-gnu',os.environ.get('LD_LIBRARY_PATH','')])
import dill
import h5py
import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from square_dp_metrics import task_metrics

DATA=Path('/home/zxiao93/Documents/Datasets/robomimic/square/ph')
BANK=ROOT/'results/kubm_square_all200/eval/20260916_230541'
RUN=ROOT/'results/official_dp_square_20260922'
EVIDENCE=ROOT/'experiment_logs/2026-09-22/official_dp_square'
ROBOT=['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
RGB=['agentview_image','robot0_eye_in_hand_image']
OmegaConf.register_new_resolver('eval',eval,replace=True)


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp.json');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def decode(v):
    if isinstance(v,dict):
        if v.get('__ndarray__'):return np.frombuffer(v['data'],dtype=v['dtype']).reshape(v['shape']).copy()
        return {k:decode(x) for k,x in v.items()}
    return v


class Sim:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='official_square_');self.log=open(Path(self.tmp.name)/'worker.log','w+')
        self.listener=Listener(str(Path(self.tmp.name)/'socket'),family='AF_UNIX',authkey=b'dp-square-local')
        self.process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('official_dp_sim_worker.py')),self.listener.address],stdout=self.log,stderr=self.log,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        self.listener._listener._socket.settimeout(60)
        self.conn=self.listener.accept();self.send('init',dataset_root=str(DATA));self.recv()
    def send(self,op,**kw):self.conn.send(dict(operation=op,**kw))
    def recv(self):
        if not self.conn.poll(90):raise TimeoutError('Simulator timeout')
        r=decode(self.conn.recv())
        if 'error' in r:raise RuntimeError(r['error'])
        return r
    def close(self):
        if hasattr(self,'conn'):self.conn.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        self.listener.close();self.log.close();self.tmp.cleanup()


def obs_feature(ob):
    return {**{k:ob[k] for k in ROBOT},**{k:np.moveaxis(ob[k],-1,0).astype('f4')/255. for k in RGB}}


def batched_sample(self,condition_data,condition_mask,local_cond=None,global_cond=None,generator=None,**kwargs):
    """Upstream sampling with an independent RNG per episode; unchanged scheduler.step."""
    gens=self._batch_generators
    assert len(gens)==len(condition_data)
    sample=torch.cat([torch.randn(condition_data[i:i+1].shape,device=condition_data.device,dtype=condition_data.dtype,generator=g) for i,g in enumerate(gens)],0)
    self.noise_scheduler.set_timesteps(self.num_inference_steps)
    for t in self.noise_scheduler.timesteps:
        sample[condition_mask]=condition_data[condition_mask]
        pred=self.model(sample,t,local_cond=local_cond,global_cond=global_cond)
        sample=torch.cat([self.noise_scheduler.step(pred[i:i+1],t,sample[i:i+1],generator=g,**kwargs).prev_sample for i,g in enumerate(gens)],0)
    sample[condition_mask]=condition_data[condition_mask]
    return sample


def load(run):
    path=RUN/'checkpoints'/f'train_{run}.ckpt';checksum=sha(path)
    payload=torch.load(path,map_location='cpu',pickle_module=dill)
    cfg=payload['cfg'];OmegaConf.resolve(cfg)
    assert cfg.task.abs_action and cfg.training.use_ema
    assert cfg.policy.num_inference_steps==100 and cfg.n_obs_steps==2 and cfg.n_action_steps==8
    policy=hydra.utils.instantiate(cfg.policy)
    state=payload['state_dicts']['ema_model'];policy.load_state_dict(state,strict=True)
    epoch=dill.loads(payload['pickles']['epoch'])
    info=dict(run=run,training_seed=int(cfg.training.seed),epoch=int(epoch),checkpoint=str(path),checkpoint_sha256=checksum,
        checkpoint_selection='Published named score-selected checkpoint; chosen before LaDiWM evaluation',
        config=OmegaConf.to_container(cfg,resolve=True),ema=True,
        vision_state_tensors=sum(k.startswith('obs_encoder.') for k in state),
        vision_parameter_count=sum(p.numel() for p in policy.obs_encoder.parameters()),
        diffusion_parameter_count=sum(p.numel() for p in policy.model.parameters()),
        image_encoder='Jointly trained random-init ResNet18 + SpatialSoftmax, two cameras, GroupNorm; EMA encoder and diffusion weights loaded strictly',
        training_wall_seconds=None,training_cost_note='External checkpoint reused; no new training. Published config/log excerpt does not provide measured trainer wall time.',
        policy_loader_sha256=sha(Path(__file__)),sim_worker_sha256=sha(Path(__file__).with_name('official_dp_sim_worker.py')),
        dp_git_commit=subprocess.check_output(['git','-C',str(DP),'rev-parse','HEAD'],text=True).strip(),
        python_executable=sys.executable,working_directory=os.getcwd(),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        evaluation_protocol='LaDiWM Eval20/Random50 state vectors in native official robosuite1.2.0 scene (joint order and state equality checked; original1.4.1 XML not loaded); controller switched to absolute mode for official10D output; 6D rotation decoded using upstream RotationTransformer; stop on first success, max400 controls at20Hz; independent seed42+i per episode.',
        training_split_limit='Official data uses val_ratio0.02 seed42, not LaDiWM180/20 mask. Eval20 is reset-aligned, not independently held out from this checkpoint.')
    save(EVIDENCE/'artifacts'/f'train_{run}_provenance.json',info)
    del payload,state;gc.collect();policy.cuda().eval()
    return policy,info


@torch.no_grad()
def evaluate(a):
    start=time.perf_counter();torch.set_num_threads(1)
    folder=EVIDENCE/f'train_{a.run}'/a.split
    if a.limit:folder=EVIDENCE/'smoke'/f'train_{a.run}'/a.split
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/'results.json'
    if output.exists() and json.loads(output.read_text())['complete']:return
    policy,info=load(a.run);rot=RotationTransformer('axis_angle','rotation_6d')
    with h5py.File(DATA/'source/low_dim_v141.hdf5') as f:names=[n.decode() for n in f['mask/valid'][:]]
    count=20 if a.split=='valid' else 50
    if a.limit:count=min(count,a.limit)
    completed=[];batch_times=[];batch_sizes=[];sims=[];slots={};next_episode=0;parity_checked=False;warmup_ms=[]
    bank_hashes={f'initial_{i:03d}.npz':sha(BANK/f'initial_{i:03d}.npz') for i in range(50)}
    torch.cuda.reset_peak_memory_stats()
    def report(complete=False):
        r=dict(complete=complete,smoke=bool(a.limit),run=a.run,training_seed=info['training_seed'],checkpoint_epoch=info['epoch'],
            split=a.split,successes=sum(e['success'] for e in completed),completed_episodes=len(completed),requested_episodes=count,
            success_rate=float(np.mean([e['success'] for e in completed])) if completed else None,
            episodes=sorted(completed,key=lambda e:e['episode']),checkpoint_sha256=info['checkpoint_sha256'],
            training_wall_seconds=None,new_training_seconds=0,task_makespan=task_metrics(completed),
            evaluator_wall_seconds=time.perf_counter()-start,episode_loop_seconds=sum(e['seconds'] for e in completed),
            sampling_total_seconds=sum(batch_times)/1000,batch_size_limit=a.batch,batch_sizes=batch_sizes,
            sampling_batch_mean_ms=float(np.mean(batch_times)) if batch_times else None,
            sampling_batch_p95_ms=float(np.percentile(batch_times,95)) if batch_times else None,
            amortized_sampling_ms_per_episode_call=sum(batch_times)/sum(batch_sizes) if batch_sizes else None,
            single_env_warm_sampling_ms=warmup_ms,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            batch1_upstream_sampler_parity=parity_checked,seed_protocol='independent torch CUDA generator42+episode index; same seeds across models and eval sets',
            initial_bank_sha256=bank_hashes if a.split=='random_saved' else None,validation_demo_names=names if a.split=='valid' else None,
            makespan_wall_note='Includes shared batched inference and waiting for other simulator slots, plus video. Simulated success steps/20 is the task duration metric. Summed episode wall overlaps and is not total compute.')
        save(output,r);return r
    try:
        for _ in range(min(a.batch,count)):sims.append(Sim())
        while next_episode<count or slots:
            for si,sim in enumerate(sims):
                if si in slots or next_episode>=count:continue
                i=next_episode;next_episode+=1;initial=None
                if a.split=='random_saved':
                    with np.load(BANK/f'initial_{i:03d}.npz') as z:initial={k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
                sim.send('reset',demo_name=names[i] if a.split=='valid' else None,initial_state=initial,image=True,video=i<3)
                ob=sim.recv();assert ob.pop('absolute_controller') and not ob['success']
                runtime=ob.pop('runtime');assert runtime['python']==sys.executable and runtime['robosuite']=='1.2.0'
                restored=ob.pop('initial_state');np.savez_compressed(folder/f'initial_{i:03d}.npz',**restored)
                if a.split=='valid':
                    with h5py.File(DATA/'source/low_dim_v141.hdf5') as f:expected=f['data'][names[i]]['states'][0]
                else:expected=initial['states']
                np.testing.assert_allclose(restored['states'],expected,rtol=0,atol=1e-10)
                f=obs_feature(ob)
                if not parity_checked:
                    test={k:torch.as_tensor(np.stack([v,v])[None],device='cuda') for k,v in f.items()}
                    policy.kwargs['generator']=torch.Generator(device='cuda').manual_seed(987)
                    original=policy.predict_action(test)['action_pred'];del policy.kwargs['generator']
                    policy.conditional_sample=types.MethodType(batched_sample,policy)
                    policy._batch_generators=[torch.Generator(device='cuda').manual_seed(987)]
                    actual=policy.predict_action(test)['action_pred']
                    torch.testing.assert_close(actual,original,rtol=0,atol=0);parity_checked=True
                    for j in range(3):
                        policy._batch_generators=[torch.Generator(device='cuda').manual_seed(990+j)]
                        torch.cuda.synchronize();t=time.perf_counter();policy.predict_action(test);torch.cuda.synchronize()
                        warmup_ms.append(1000*(time.perf_counter()-t))
                slots[si]=dict(i=i,ob=ob,restore_error=float(np.max(np.abs(restored['states']-expected))),runtime=runtime,history=deque([f,f],maxlen=2),generator=torch.Generator(device='cuda').manual_seed(42+i),
                    actions=[],raw=[],robot=[],replans=[],seconds_start=time.perf_counter(),
                    writer=imageio.get_writer(folder/f'episode_{i:03d}.mp4',fps=20) if i<3 else None)
            active=list(slots);batch={k:torch.as_tensor(np.stack([np.stack([x[k] for x in slots[si]['history']]) for si in active]),device='cuda') for k in ROBOT+RGB}
            policy._batch_generators=[slots[si]['generator'] for si in active]
            torch.cuda.synchronize();tick=time.perf_counter();raw=policy.predict_action(batch)['action'].cpu().numpy();torch.cuda.synchronize()
            elapsed=1000*(time.perf_counter()-tick);batch_times.append(elapsed);batch_sizes.append(len(active))
            assert raw.shape==(len(active),8,10) and np.isfinite(raw).all()
            env_action=np.concatenate([raw[...,:3],rot.inverse(raw[...,3:9]),raw[...,9:]],axis=-1)
            for si in active:slots[si]['replans'].append(len(slots[si]['actions']))
            for step in range(8):
                current=[si for si in active if si in slots]
                for si in current:
                    s=slots[si];bi=active.index(si)
                    if s['writer']:s['writer'].append_data(s['ob']['rgb'])
                    s['robot'].append(np.concatenate([s['ob'][k] for k in ROBOT]));s['actions'].append(env_action[bi,step]);s['raw'].append(raw[bi,step])
                    sims[si].send('step',action=env_action[bi,step],image=True,video=s['i']<3)
                for si in current:
                    s=slots[si];ob=sims[si].recv();s['ob']=ob;s['history'].append(obs_feature(ob))
                    if ob['success'] or ob['done'] or len(s['actions'])>=a.max_steps:
                        wall=time.perf_counter()-s['seconds_start']
                        if s['writer']:s['writer'].append_data(ob['rgb']);s['writer'].close()
                        np.testing.assert_array_equal(s['replans'],np.arange(0,len(s['actions']),8))
                        np.savez_compressed(folder/f'trajectory_{s["i"]:03d}.npz',actions=s['actions'],raw_absolute10=s['raw'],robot=s['robot'],replan_steps=s['replans'])
                        e=dict(episode=s['i'],demo_name=names[s['i']] if a.split=='valid' else None,success=bool(ob['success']),steps=len(s['actions']),seconds=wall,execution_wall_seconds=wall,
                            initial_state_max_abs_error=s['restore_error'],absolute_controller_verified=True,simulator_runtime=s['runtime'],video=f'episode_{s["i"]:03d}.mp4' if s['i']<3 else None)
                        completed.append(e);del slots[si];report();print(json.dumps(e),flush=True)
        report(not a.limit and a.max_steps==400)
    finally:
        for s in slots.values():
            if s['writer']:s['writer'].close()
        for sim in sims:sim.close()
    result=report(not a.limit and a.max_steps==400);print('COMPLETE',a.run,a.split,result['successes'],count,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=int,required=True);p.add_argument('--split',choices=['valid','random_saved'],required=True)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--limit',type=int);p.add_argument('--max-steps',type=int,default=400)
    evaluate(p.parse_args())
