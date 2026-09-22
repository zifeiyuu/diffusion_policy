"""Paired Eval20/Random50 rollouts of final EMA policies, with live causal observations."""
import argparse
from collections import deque
import os
import random
import subprocess
import tempfile
import time
from multiprocessing.connection import Listener

from square_dp_data import *
import imageio.v2 as imageio
from square_dp_metrics import task_metrics


def portable_decode(value):
    if isinstance(value, dict):
        if value.get('__ndarray__'):
            return np.frombuffer(value['data'], dtype=value['dtype']).reshape(value['shape']).copy()
        return {k:portable_decode(v) for k,v in value.items()}
    return value


class Simulator:
    def __init__(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dp_square_')
        self.log=open(Path(self.tmp.name)/'worker.log','w+')
        self.listener=Listener(str(Path(self.tmp.name)/'socket'),family='AF_UNIX',authkey=b'dp-square-local')
        self.process=subprocess.Popen([os.environ.get('DP_SIM_PYTHON','/home/zxiao93/anaconda3/envs/kguide/bin/python'),
            str(Path(__file__).with_name('square_dp_sim_worker.py')),self.listener.address],
            stdout=self.log,stderr=self.log,env={**os.environ,'MUJOCO_GL':'egl'})
        self.listener._listener._socket.settimeout(60)
        try:
            self.conn=self.listener.accept();self.request('init',dataset_root=str(DATA))
        except BaseException:
            self.log.flush();self.log.seek(0);print(self.log.read());self.close();raise
    def request(self,op,**kw):
        self.conn.send(dict(operation=op,**kw))
        if not self.conn.poll(120):raise TimeoutError('Simulator timeout')
        r=portable_decode(self.conn.recv())
        if 'error' in r:raise RuntimeError(r['error'])
        return r
    def close(self):
        if hasattr(self,'conn'):self.conn.close()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        self.listener.close();self.log.close();self.tmp.cleanup()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def feature(obs,previous,modality,ae=None):
    f={k:np.asarray(obs[k],dtype='f4') for k in ROBOT}
    if modality=='image':
        f.update({k:np.moveaxis(obs[k],-1,0).astype('f4')/255. for k in RGB})
    elif modality=='point_ae':
        assert ae is not None
        f['point_ae']=ae(torch.as_tensor(obs['points'][None],device=next(ae.parameters()).device))[0].cpu().numpy()
    else:
        p=obs['points']/np.array([640,480],dtype='f4')
        prev=p if previous is None else previous/np.array([640,480],dtype='f4')
        f['point_flow']=np.r_[p.reshape(-1),(p-prev).reshape(-1)].astype('f4')
    return f


@torch.no_grad()
def evaluate(a):
    began=time.perf_counter();torch.set_num_threads(4)
    output=Path(a.run);checkpoint=output/'final_ema.pt';checkpoint_hash=digest(checkpoint)
    report=output/a.split;report.mkdir(exist_ok=True)
    result_path=report/'results.json'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['complete'] and old['checkpoint_sha256']==checkpoint_hash and not a.limit:return old
    saved=torch.load(checkpoint,map_location='cpu');cfg=saved['config']
    assert saved['epoch']==cfg['epochs']-1
    assert a.limit or not cfg['smoke']
    model=make_policy(cfg['modality']);model.load_state_dict(saved['model']);model.cuda().eval();del saved
    ae=None
    if cfg['modality']=='point_ae':
        from square_point_ae import FrozenPointAE,ae_sha
        assert cfg['point_ae']['ae_sha256']==ae_sha()
        ae=FrozenPointAE().cuda()
    manifest=json.loads((OUT/'cache/manifest.json').read_text())
    assert digest(OUT/'cache/manifest.json')==cfg['manifest_sha256']
    if a.split=='random_saved':
        for name,sha in manifest['random_bank'].items():assert digest(BANK/name)==sha
    names=manifest['splits']['valid'];expected=20 if a.split=='valid' else 50
    count=min(a.limit,expected) if a.limit else expected
    records=[];ae_ms=[];sample_ms=[];sim_ms=[];max_action_excess=0.;torch.cuda.reset_peak_memory_stats()
    def save(complete=False):
        result=dict(complete=complete,smoke=bool(a.limit),modality=cfg['modality'],training_seed=cfg['seed'],
            split=a.split,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,checkpoint_epoch=cfg['epochs']-1,
            successes=sum(x['success'] for x in records),completed_episodes=len(records),requested_episodes=expected,
            success_rate=float(np.mean([x['success'] for x in records])) if records else None,
            max_action_roundoff_clipped=max_action_excess,
            task_makespan=task_metrics(records),
            frozen_ae_sha256=cfg.get('point_ae',{}).get('ae_sha256'),
            frozen_ae_calls=len(ae_ms),frozen_ae_total_seconds=float(np.sum(ae_ms))/1000,
            frozen_ae_mean_ms=float(np.mean(ae_ms)) if ae_ms else None,
            frozen_ae_p95_ms=float(np.percentile(ae_ms,95)) if ae_ms else None,
            combined_policy_compute_seconds=(float(np.sum(sample_ms))+float(np.sum(ae_ms)))/1000,
            sampling_total_seconds=float(np.sum(sample_ms))/1000,
            sampling_calls=len(sample_ms),simulator_total_seconds=float(np.sum(sim_ms))/1000,
            evaluator_source_sha256=digest(Path(__file__)),metrics_source_sha256=digest(Path(__file__).with_name('square_dp_metrics.py')),
            episode_seed='42 + episode index after restoring state',horizon=400,sampler='DDPM100',
            initial_bank=str(BANK) if a.split=='random_saved' else 'HDF5 mask/valid',
            episodes=records,episode_loop_seconds=sum(x['seconds'] for x in records),
            evaluator_wall_seconds=time.perf_counter()-began,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            sampling_mean_ms=float(np.mean(sample_ms)) if sample_ms else None,
            sampling_p95_ms=float(np.percentile(sample_ms,95)) if sample_ms else None,
            simulator_mean_ms=float(np.mean(sim_ms)) if sim_ms else None)
        write_json(result_path,result);return result
    with Simulator() as sim:
        for i in range(count):
            initial=None
            if a.split=='random_saved':
                with np.load(BANK/f'initial_{i:03d}.npz',allow_pickle=False) as f:
                    initial={k:f[k].item() if f[k].ndim==0 else f[k].copy() for k in f.files}
            video=i<3;is_image=cfg['modality']=='image'
            obs=sim.request('reset',demo_name=names[i] if a.split=='valid' else None,
                initial_state=initial,image=is_image,video=video)
            restored=obs.pop('initial_state');assert not obs['success']
            np.savez_compressed(report/f'initial_{i:03d}.npz',**restored)
            torch.manual_seed(42+i);np.random.seed(42+i);random.seed(42+i)
            history=deque(maxlen=2);actions=deque();previous=None;trace=[];poses=[];replans=[];inputs=[]
            writer=imageio.get_writer(report/f'episode_{i:03d}.mp4',fps=20) if video else None
            tick=time.perf_counter();success=False
            try:
                for t in range(a.max_steps):
                    if ae is not None:torch.cuda.synchronize()
                    ft=time.perf_counter()
                    f=feature(obs,previous,cfg['modality'],ae);previous=obs['points'].copy()
                    if ae is not None:
                        torch.cuda.synchronize();ae_ms.append(1000*(time.perf_counter()-ft))
                    if not history:history.append(f)
                    history.append(f);poses.append(np.concatenate([obs[k] for k in ROBOT]))
                    if writer:writer.append_data(obs['rgb'])
                    if not actions:
                        batch={k:torch.as_tensor(np.stack([x[k] for x in history])[None],device='cuda') for k in f}
                        if cfg['modality']=='point_flow': batch['point_flow'][:,0,512:] = 0.
                        torch.cuda.synchronize();ts=time.perf_counter()
                        prediction=model.predict_action(batch)['action'][0].cpu().numpy()
                        torch.cuda.synchronize();sample_ms.append(1000*(time.perf_counter()-ts))
                        assert prediction.shape==(8,7) and np.isfinite(prediction).all()
                        excess=max(0.,float(np.max(np.abs(prediction)))-1.)
                        assert excess<1e-3, excess
                        max_action_excess=max(max_action_excess,excess)
                        # diffusers0.11 DDPM terminal coefficients can exceed1 by ~1.4e-5.
                        # Matches the OSC controller's existing [-1,1] input clipping.
                        prediction=np.clip(prediction,-1.,1.)
                        actions.extend(prediction);replans.append(t)
                        if not is_image:inputs.append(batch[cfg['modality']].cpu().numpy()[0])
                    action=actions.popleft();trace.append(action)
                    ts=time.perf_counter();obs=sim.request('step',action=action,image=is_image,video=video)
                    sim_ms.append(1000*(time.perf_counter()-ts));success=obs['success']
                    if success or obs['done']:break
                execution_wall_seconds=time.perf_counter()-tick
                if writer:writer.append_data(obs['rgb'])
            finally:
                if writer:writer.close()
            np.testing.assert_array_equal(replans,np.arange(0,len(trace),8))
            np.savez_compressed(report/f'trajectory_{i:03d}.npz',actions=np.asarray(trace),robot=np.asarray(poses),
                replan_steps=replans,visual_inputs=np.asarray(inputs),visual_input_kind=cfg["modality"],
                point_flow_inputs=np.asarray(inputs) if cfg["modality"]=="point_flow" else np.empty((0,)))
            record=dict(episode=i,demo_name=names[i] if a.split=='valid' else None,success=bool(success),
                steps=len(trace),seconds=time.perf_counter()-tick,execution_wall_seconds=execution_wall_seconds,
                task_sim_seconds=len(trace)/20,
                termination="success" if success else "environment_done" if obs["done"] else "horizon",
                video=f'episode_{i:03d}.mp4' if video else None,initial_state_verified=True)
            records.append(record);save();print(json.dumps(record),flush=True)
    result=save(complete=len(records)==expected and not a.limit and a.max_steps==400)
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    p.add_argument('--split',choices=['valid','random_saved'],required=True)
    p.add_argument('--limit',type=int);p.add_argument('--max-steps',type=int,default=400)
    evaluate(p.parse_args())
