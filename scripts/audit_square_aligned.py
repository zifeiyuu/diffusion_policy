"""Check split isolation, upstream sampling, causal flow, and live observation parity."""
from square_dp_data import *
from eval_square_aligned import Simulator,feature
from diffusion_policy.common.sampler import create_indices


def main():
    info=prepare();train=SquareDataset('point_flow','train',info);valid=SquareDataset('point_flow','valid',info)
    assert not set(train.rows)&set(valid.rows)
    ends=np.array(info['ends']);mask=np.array([n in info['splits']['train'] for n in info['names']])
    upstream=create_indices(ends,16,mask,pad_before=1,pad_after=7)
    assert len(upstream)==len(train)
    for i,(start,length,t) in enumerate(train.samples):
        bs,be,ss,se=upstream[i]
        arr=train.arrays['action'][bs:be]
        target=np.concatenate([np.repeat(arr[:1],ss,axis=0),arr,np.repeat(arr[-1:],16-se,axis=0)])
        batch=train[i]
        np.testing.assert_array_equal(batch['action'].numpy(),target)
        assert t>=0 and t<length
        assert not batch['obs']['point_flow'][0,512:].any()
        if t==0:assert not batch['obs']['point_flow'][:,-512:].any()
    checks=[]
    with Simulator() as sim:
        for name in ['demo_0','demo_103','demo_5']:
            obs=sim.request('reset',demo_name=name,image=True)
            with h5py.File(DATA/f'gt_geometry/episodes/{name}.hdf5') as g,h5py.File(DATA/'source/image_v141.hdf5') as im:
                np.testing.assert_allclose(obs['points'],g['points_xy'][0],atol=2e-4,rtol=0)
                for k in ROBOT:np.testing.assert_allclose(obs[k],im['data'][name]['obs'][k][0],atol=1e-5,rtol=0)
                images={}
                for k in RGB:
                    expected=im['data'][name]['obs'][k][0].astype('f4');actual=obs[k].astype('f4')
                    mse=float(np.mean((actual-expected)**2));flipped=float(np.mean((actual[::-1]-expected)**2))
                    assert mse<flipped and mse<100,(name,k,mse,flipped)
                    images[k]=dict(mse=mse,vertical_flip_mse=flipped)
                f=feature(obs,None,'point_flow')['point_flow']
                index=info['names'].index(name);start=0 if index==0 else info['ends'][index-1]
                np.testing.assert_allclose(f,train.arrays['point_flow'][start],atol=1e-6,rtol=0)
            checks.append(dict(demo=name,images=images,point_and_robot_parity=True))
        # Independently restore first and last bank entries as well.
        for i in [0,49]:
            with np.load(BANK/f'initial_{i:03d}.npz') as f:
                initial={k:f[k].item() if f[k].ndim==0 else f[k].copy() for k in f.files}
            obs=sim.request('reset',initial_state=initial)
            np.testing.assert_allclose(obs['initial_state']['states'],initial['states'],rtol=0,atol=1e-10)
    result=dict(passed=True,train_demos=180,validation_demos=20,train_samples=len(train),valid_samples=len(valid),
        every_training_action_window_matches_upstream=True,normalization_rows_disjoint=True,
        point_flow='backward difference with zero t0, never stored forward flow',live_checks=checks,
        random_bank_endpoints_verified=True,manifest_sha256=digest(OUT/'cache/manifest.json'))
    write_json(REPORT/'artifacts/audit.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
