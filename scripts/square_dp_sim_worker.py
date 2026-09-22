"""Robosuite 1.4.1 worker, using LaDiWM geometry and identical reset states."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import sys
import traceback
from pathlib import Path
from multiprocessing.connection import Client
import h5py
import numpy as np
sys.path.insert(0,os.environ.get('DP_LADIWM_ROOT','/home/zxiao93/Documents/LaDiWM'))
from ladiwm.kubm.geometry_source import geometry,surface_points,project
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite


def portable(value):
    # NumPy 2 simulator arrays must not pickle NumPy-2-only module paths into NumPy 1 training.
    if isinstance(value, np.ndarray):
        return dict(__ndarray__=True, dtype=value.dtype.str, shape=value.shape, data=value.tobytes())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k:portable(v) for k,v in value.items()}
    return value


def main():
    conn=Client(sys.argv[1],family='AF_UNIX',authkey=b'dp-square-local');env=None
    try:
        while True:
            try:d=conn.recv()
            except EOFError:break
            try:
                op=d['operation']
                if op=='init':
                    root=Path(d['dataset_root']);np.random.seed(42)
                    with h5py.File(root/'source/low_dim_v141.hdf5') as f:meta=json.loads(f['data'].attrs['env_args'])
                    assert robosuite.__version__==meta['env_version']=='1.4.1'
                    ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'low_dim':['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object']}})
                    env=EnvUtils.create_env_from_metadata(meta,render=False,render_offscreen=True,use_image_obs=False)
                    anchors=np.load(root/'gt_geometry/anchors_body.npy');result={}
                elif op in ['reset','step']:
                    if op=='reset':
                        initial=d.get('initial_state')
                        if initial is None:
                            with h5py.File(root/'source/low_dim_v141.hdf5') as f:
                                demo=f['data'][d['demo_name']];initial=dict(model=demo.attrs['model_file'],states=demo['states'][0])
                                if 'ep_meta' in demo.attrs:initial['ep_meta']=demo.attrs['ep_meta']
                        obs=env.reset_to(initial)
                        np.testing.assert_allclose(env.get_state()['states'],initial['states'],rtol=0,atol=1e-10)
                        body,_,boxes=geometry(env.env.sim);np.testing.assert_array_equal(surface_points(boxes),anchors)
                        reward,done=0.,False
                    else:
                        action=np.asarray(d['action'],dtype='f4')
                        assert action.shape==(7,) and np.isfinite(action).all() and abs(action).max()<=1.00001
                        obs,reward,done,_=env.step(action)
                    result={k:obs[k].astype('f4') for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}
                    result.update(points=project(env.env.sim,body,anchors)[0].astype('f4'),
                        reward=float(reward),done=bool(done),success=bool(env.is_success()['task']))
                    if d.get('image',False):
                        for camera in ['agentview','robot0_eye_in_hand']:
                            result[camera+'_image']=env.env.sim.render(width=84,height=84,camera_name=camera)[::-1].copy()
                    if d.get('video',False):
                        result['rgb']=env.env.sim.render(width=640,height=480,camera_name='agentview')[::-1].copy()
                    if op=='reset':result['initial_state']=env.get_state()
                else:raise ValueError(op)
                conn.send(portable(result))
            except Exception:conn.send({'error':traceback.format_exc()})
    finally:
        if env is not None:env.env.close()
        conn.close()


if __name__=='__main__':main()
