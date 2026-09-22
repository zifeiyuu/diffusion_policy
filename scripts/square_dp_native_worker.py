"""Native official simulator for retrained delta-action image and PointAE DP."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from multiprocessing.connection import Client
import h5py
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
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


def project_points(sim,anchors):
    body=sim.model.body_name2id('SquareNut_main')
    world=anchors@sim.data.body_xmat[body].reshape(3,3).T+sim.data.body_xpos[body]
    cam=sim.model.camera_name2id('agentview')
    rotation=sim.data.cam_xmat[cam].reshape(3,3)@np.diag([1.,-1.,-1.])
    local=(world-sim.data.cam_xpos[cam])@rotation
    f=240./np.tan(sim.model.cam_fovy[cam]*np.pi/360.)
    return local[:,:2]/local[:,2,None]*f+np.array([320.,240.])


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
                    assert robosuite.__version__=='1.2.0'
                    ObsUtils.initialize_obs_utils_with_obs_specs({'obs':{'low_dim':['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos','object']}})
                    meta['env_kwargs']['controller_configs']['control_delta']=True
                    env=EnvUtils.create_env_from_metadata(meta,render=False,render_offscreen=True,use_image_obs=False)
                    anchors=np.load(root/'gt_geometry/anchors_body.npy');result={}
                elif op in ['reset','step']:
                    if op=='reset':
                        initial=d.get('initial_state')
                        if initial is None:
                            with h5py.File(root/'source/low_dim_v141.hdf5') as f:
                                demo=f['data'][d['demo_name']];initial=dict(model=demo.attrs['model_file'],states=demo['states'][0])
                                if 'ep_meta' in demo.attrs:initial['ep_meta']=demo.attrs['ep_meta']
                        # Restore the paired state into the native official scene.
                        # Modern saved XML cannot be loaded by official MuJoCo 2.1.
                        xml=ET.fromstring(initial['model'])
                        names=[e.attrib['name'] for e in xml.find('worldbody').iter('joint')]
                        assert names==list(env.env.sim.model.joint_names),(names,env.env.sim.model.joint_names)
                        env.reset()
                        obs=env.reset_to({'states':initial['states']})
                        np.testing.assert_allclose(env.get_state()['states'],initial['states'],rtol=0,atol=1e-10)
                        reward,done=0.,False
                    else:
                        action=np.asarray(d['action'],dtype='f4')
                        assert action.shape==(7,) and np.isfinite(action).all()
                        assert env.env.robots[0].controller.use_delta
                        assert abs(action).max()<=1.00001
                        obs,reward,done,_=env.step(action)
                    result={k:obs[k].astype('f4') for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos']}
                    result['points']=project_points(env.env.sim,anchors).astype('f4')
                    result.update(reward=float(reward),done=bool(done),success=bool(env.is_success()['task']))
                    if d.get('image',False):
                        for camera in ['agentview','robot0_eye_in_hand']:
                            result[camera+'_image']=env.env.sim.render(width=84,height=84,camera_name=camera)[::-1].copy()
                    if d.get('video',False):
                        result['rgb']=env.env.sim.render(width=640,height=480,camera_name='agentview')[::-1].copy()
                    if op=='reset':
                        result['initial_state']=env.get_state()
                        result['delta_controller']=env.env.robots[0].controller.use_delta
                        result['runtime']=dict(python=sys.executable,robosuite=robosuite.__version__,robosuite_path=robosuite.__file__,joint_names=names,scene='native official robosuite1.2.0; state vector restored from LaDiWM')

                else:raise ValueError(op)
                conn.send(portable(result))
            except Exception:conn.send({'error':traceback.format_exc()})
    finally:
        if env is not None:env.env.close()
        conn.close()


if __name__=='__main__':main()
