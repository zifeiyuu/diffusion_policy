"""Check paired reset vectors and native official scene placement before evaluation."""
import sys
import xml.etree.ElementTree as ET
import json
import numpy as np
import h5py
from eval_official_dp_square import Sim,DATA,BANK,EVIDENCE,save


def audit():
    sim=Sim();records=[]
    try:
        with h5py.File(DATA/'source/low_dim_v141.hdf5') as f:
            names=[n.decode() for n in f['mask/valid'][:]]
            for split,count in [('valid',20),('random_saved',50)]:
                for i in range(count):
                    if split=='valid':
                        demo=f['data'][names[i]];initial={'model':demo.attrs['model_file'],'states':demo['states'][0]}
                    else:
                        with np.load(BANK/f'initial_{i:03d}.npz') as z:initial={k:z[k].item() if z[k].ndim==0 else z[k].copy() for k in z.files}
                    sim.send('reset',initial_state=initial,image=False,video=False);r=sim.recv()
                    error=float(np.max(np.abs(r['initial_state']['states']-initial['states'])))
                    assert error<=1e-10
                    a,b=ET.fromstring(initial['model']),ET.fromstring(r['initial_state']['model'])
                    paired=[]
                    for tag,names_to_check in [('body',['table','peg1','peg2','robot0_base']),('camera',['agentview','robot0_eye_in_hand'])]:
                        for name in names_to_check:
                            x=a.find(f'.//{tag}[@name="{name}"]');y=b.find(f'.//{tag}[@name="{name}"]')
                            assert x is not None and y is not None
                            for key,default in [('pos','0 0 0'),('quat','1 0 0 0')]:
                                xv=np.fromstring(x.attrib.get(key,default),sep=' ');yv=np.fromstring(y.attrib.get(key,default),sep=' ')
                                if key=='quat':
                                    xv=xv/np.linalg.norm(xv);yv=yv/np.linalg.norm(yv)
                                    if np.dot(xv,yv)<0:yv=-yv
                                np.testing.assert_allclose(xv,yv,atol=1e-5,rtol=0)
                            paired.append(name)
                    robot_error=None
                    if split=='valid':
                        robot_error=max(float(np.max(np.abs(r[k]-demo['obs'][k][0]))) for k in ['robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'])
                        assert robot_error<1e-5
                    records.append(dict(split=split,episode=i,state_error=error,robot_error=robot_error,matched_scene_poses=paired,runtime=r['runtime']))
                    print(split,i,error,flush=True)
        save(EVIDENCE/'artifacts/native_official_reset_audit.json',dict(passed=True,episodes=records,scope='Same state vectors/joint order and fixed table/peg/robot-base/camera poses; native official1.2 scene and assets, not pixel- or dynamics-identical to LaDiWM1.4.1.'))
    finally:sim.close()
if __name__=='__main__':audit()
