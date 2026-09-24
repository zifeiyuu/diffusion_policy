"""Batched evaluation of retrained100-epoch image/PointAE DP, native official simulator."""
import argparse
from collections import deque
import gc
import hashlib
import json
import os
from pathlib import Path
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
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from square_dp_metrics import task_metrics
from square_upstream_adapters import config
import hydra
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from square_point_ae import FrozenPointAE,ae_sha

SIM_PYTHON='/home/zxiao93/anaconda3/envs/dp_eval141/bin/python'
DATA=Path('/home/zxiao93/Documents/Datasets/robomimic/square/ph')
BANK=ROOT/'results/kubm_square_all200/eval/20260916_230541'
ROBOT=['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']
RGB=['agentview_image','robot0_eye_in_hand_image']


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
        self.process=subprocess.Popen([SIM_PYTHON,str(Path(__file__).with_name('square_upstream_worker.py')),self.listener.address],stdout=self.log,stderr=self.log,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        self.listener._listener._socket.settimeout(60)
        self.conn=self.listener.accept();self.send('init',dataset_root=str(DATA));self.recv()
    def send(self,op,**kw):self.conn.send(dict(operation=op,**kw))
    def recv(self):
        if not self.conn.poll(90):
            self.log.flush();self.log.seek(0)
            detail=self.log.read()[-4000:]
            raise TimeoutError(f'Simulator timeout: pid={self.process.pid}, exit={self.process.poll()}, worker log={detail}')
        r=decode(self.conn.recv())
        if 'error' in r:raise RuntimeError(r['error'])
        return r
    def close(self):
        if hasattr(self,'conn'):self.conn.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        self.listener.close();self.log.close();self.tmp.cleanup()


def obs_feature(ob,modality):
    extra={k:np.moveaxis(ob[k],-1,0).astype('f4')/255. for k in RGB} if modality=='image' else {'points':ob['points']}
    return {**{k:ob[k] for k in ROBOT},**extra}


def batched_sample(self,condition_data,condition_mask,local_cond=None,global_cond=None,generator=None,cond=None,**kwargs):
    """Upstream sampling with an independent RNG per episode; unchanged scheduler.step."""
    gens=self._batch_generators
    assert len(gens)==len(condition_data)
    sample=torch.cat([torch.randn(condition_data[i:i+1].shape,device=condition_data.device,dtype=condition_data.dtype,generator=g) for i,g in enumerate(gens)],0)
    self.noise_scheduler.set_timesteps(self.num_inference_steps)
    for t in self.noise_scheduler.timesteps:
        sample[condition_mask]=condition_data[condition_mask]
        pred=self.model(sample,t,cond) if getattr(self,'_dp_architecture','unet')=='transformer' else self.model(sample,t,local_cond=local_cond,global_cond=global_cond)
        sample=torch.cat([self.noise_scheduler.step(pred[i:i+1],t,sample[i:i+1],generator=g,**kwargs).prev_sample for i,g in enumerate(gens)],0)
    sample[condition_mask]=condition_data[condition_mask]
    return sample


def load(run):
    path=Path(run)/'policy.pt';payload=torch.load(path,map_location='cpu');cfg=payload['config']
    if cfg.get('down_dims') is not None:cfg['down_dims']=list(cfg['down_dims'])
    assert cfg['modality'] in ['image','point_ae'] and cfg['seed']==42 and cfg['epochs'] in [50,100] and not cfg['smoke']
    resolved,_=config(cfg['modality'],down_dims=cfg.get('down_dims'),architecture=cfg.get('architecture','unet'));policy=hydra.utils.instantiate(resolved.policy);policy.load_state_dict(payload['model'],strict=True);policy.cuda().eval()
    policy._dp_architecture=cfg.get('architecture','unet')
    if cfg['modality']=='point_ae':
        ae=FrozenPointAE().cuda().eval();original=policy.predict_action
        def with_ae(ob):
            points=ob['points'];B,T=points.shape[:2]
            data={k:ob[k] for k in ROBOT};data['point_ae_qpos']=ae(points.reshape(-1,256,2)).reshape(B,T,128)
            return original(data)
        policy.predict_action=with_ae
    info=dict(run=str(run),training_seed=42,epoch=payload['epoch'],modality=cfg['modality'],checkpoint=str(path),checkpoint_sha256=sha(path),config=cfg,
        python_executable=sys.executable,working_directory=os.getcwd(),evaluation_source_sha256=sha(Path(__file__)),worker_sha256=sha(Path(__file__).with_name('square_upstream_worker.py')))
    save(Path(run)/'evaluation_provenance.json',info)
    del payload;gc.collect()
    return policy,info


@torch.no_grad()
def evaluate(a):
    start=time.perf_counter();torch.set_num_threads(1)
    folder=Path(a.run)/a.split
    if a.limit:folder=Path(a.run)/'smoke'/a.split
    folder.mkdir(parents=True,exist_ok=True)
    output=folder/'results.json'
    if output.exists() and json.loads(output.read_text())['complete']:return
    policy,info=load(a.run);modality=info['modality'];visual_keys=RGB if modality=='image' else ['points']
    with h5py.File(DATA/'source/low_dim_v141.hdf5') as f:names=[n.decode() for n in f['mask/valid'][:]]
    count=20 if a.split=='valid' else 50
    if a.limit:count=min(count,a.limit)
    completed=[];batch_times=[];batch_sizes=[];sims=[];slots={};next_episode=0;parity_checked=False;warmup_ms=[]
    bank_hashes={f'initial_{i:03d}.npz':sha(BANK/f'initial_{i:03d}.npz') for i in range(50)}
    torch.cuda.reset_peak_memory_stats()
    def report(complete=False):
        r=dict(complete=complete,smoke=bool(a.limit),run=a.run,modality=modality,training_seed=info['training_seed'],checkpoint_epoch=info['epoch'],
            split=a.split,successes=sum(e['success'] for e in completed),completed_episodes=len(completed),requested_episodes=count,
            success_rate=float(np.mean([e['success'] for e in completed])) if completed else None,
            episodes=sorted(completed,key=lambda e:e['episode']),checkpoint_sha256=info['checkpoint_sha256'],
            training_wall_seconds=None,task_makespan=task_metrics(completed),
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
                sim.send('reset',demo_name=names[i] if a.split=='valid' else None,initial_state=initial,image=modality=='image',video=i<3)
                ob=sim.recv();assert not ob.pop('delta_controller') and not ob['success']
                runtime=ob.pop('runtime');assert runtime['python']==SIM_PYTHON and runtime['robosuite']=='1.4.1'
                restored=ob.pop('initial_state');np.savez_compressed(folder/f'initial_{i:03d}.npz',**restored)
                if a.split=='valid':
                    with h5py.File(DATA/'source/low_dim_v141.hdf5') as f:expected=f['data'][names[i]]['states'][0]
                else:expected=initial['states']
                np.testing.assert_allclose(restored['states'],expected,rtol=0,atol=1e-10)
                f=obs_feature(ob,modality)
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
            active=list(slots);batch={k:torch.as_tensor(np.stack([np.stack([x[k] for x in slots[si]['history']]) for si in active]),device='cuda') for k in ROBOT+visual_keys}
            policy._batch_generators=[slots[si]['generator'] for si in active]
            torch.cuda.synchronize();tick=time.perf_counter();raw=policy.predict_action(batch)['action'].cpu().numpy();torch.cuda.synchronize()
            elapsed=1000*(time.perf_counter()-tick);batch_times.append(elapsed);batch_sizes.append(len(active))
            assert raw.shape==(len(active),8,10) and np.isfinite(raw).all()
            rot=RotationTransformer(from_rep='axis_angle',to_rep='rotation_6d')
            env_action=np.concatenate([raw[...,:3],rot.inverse(raw[...,3:9]),raw[...,9:]],axis=-1)
            for si in active:slots[si]['replans'].append(len(slots[si]['actions']))
            for step in range(8):
                current=[si for si in active if si in slots]
                for si in current:
                    s=slots[si];bi=active.index(si)
                    if s['writer']:s['writer'].append_data(s['ob']['rgb'])
                    s['robot'].append(np.concatenate([s['ob'][k] for k in ROBOT]));s['actions'].append(env_action[bi,step]);s['raw'].append(raw[bi,step])
                    sims[si].send('step',action=env_action[bi,step],image=modality=='image',video=s['i']<3)
                for si in current:
                    s=slots[si];ob=sims[si].recv();s['ob']=ob;s['history'].append(obs_feature(ob,modality))
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
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--split',choices=['valid','random_saved'],required=True)
    p.add_argument('--batch',type=int,default=2);p.add_argument('--limit',type=int);p.add_argument('--max-steps',type=int,default=400)
    evaluate(p.parse_args())
