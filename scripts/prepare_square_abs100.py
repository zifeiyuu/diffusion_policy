"""Upstream absolute-action conversion; preserve source observations and masks."""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import h5py, numpy as np
from diffusion_policy.common.robomimic_util import RobomimicAbsoluteActionConverter
from square_dp_data import DATA, ROBOT, RGB, digest
ROOT=Path('/home/zxiao93/Documents/LaDiWM/results/dp_upstream_abs100_20260922')
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    target=ROOT/'image_abs.hdf5'; meta=ROOT/'conversion.json'
    if meta.exists() and target.exists(): return
    start=time.perf_counter();source=DATA/'source/image_v141.hdf5'
    converter=RobomimicAbsoluteActionConverter(str(DATA/'source/low_dim_v141.hdf5'))
    converter.env.reset();converter.abs_env.reset()
    ae=np.load('/home/zxiao93/Documents/LaDiWM/results/dp_shared_cache/point_ae.npy',mmap_mode='r')
    errors=[];offset=0
    # Each trajectory can resume conversion independently after an interruption.
    with h5py.File(source,'r') as src,h5py.File(target,'a') as dst:
        data=dst.require_group('data')
        for k,v in src['data'].attrs.items():data.attrs[k]=v
        if 'mask' not in dst:dst['mask']=h5py.ExternalLink(str(source),'/mask')
        for i in range(len(src['data'])):
            name=f'demo_{i}';d=src['data'][name];n=len(d['actions'])
            if name in data and data[name].attrs.get('converted',False):offset+=n;continue
            if name in data:del data[name]
            out=data.create_group(name)
            for k,v in d.attrs.items():out.attrs[k]=v
            states=d['states'][:];actions=d['actions'][:]
            absolute=converter.convert_actions(states,actions)
            # Compare delta and absolute next-state transitions on representative frames.
            for j in sorted(set([1,n//2,n-1])):
                converter.env.reset_to({'states':states[j]});converter.env.step(actions[j])
                ds=converter.env.get_state()['states']
                converter.abs_env.reset_to({'states':states[j]});converter.abs_env.step(absolute[j])
                ab=converter.abs_env.get_state()['states']
                err=float(np.max(np.abs(ds-ab)));errors.append(err)
                # Independent controller nullspace initialization can change next states.
                # Verify the converted OSC targets directly; record rollout discrepancy.
                dc=converter.env.env.robots[0].controller;ac=converter.abs_env.env.robots[0].controller
                np.testing.assert_allclose(dc.goal_pos,ac.goal_pos,rtol=0,atol=1e-6)
                np.testing.assert_allclose(dc.goal_ori,ac.goal_ori,rtol=0,atol=1e-6)
            out.create_dataset('actions',data=absolute)
            out['states']=h5py.ExternalLink(str(source),f'/data/{name}/states')
            obs=out.create_group('obs')
            for key in ROBOT+RGB:obs[key]=h5py.ExternalLink(str(source),f'/data/{name}/obs/{key}')
            obs.create_dataset('point_ae_qpos',data=ae[offset:offset+n])
            out.attrs['converted']=True;offset+=n;dst.flush()
            print('Converted',name,flush=True)
    link=ROOT/'point_ae_abs.hdf5'
    if not link.exists():link.symlink_to(target.name)
    meta.write_text(json.dumps(dict(source=str(source),source_sha256=digest(source),converter='upstream RobomimicAbsoluteActionConverter.convert_actions',action='absolute xyz + axis-angle + gripper; upstream dataset converts to rotation6d',max_checked_next_state_error=max(errors) if errors else None,checked_transitions=len(errors),wall_seconds=time.perf_counter()-start,python=sys.executable,normalizer='unchanged upstream: statistics over replay buffer'),indent=2)+'\n')
if __name__=='__main__':main()
