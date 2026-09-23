"""Preflight both adapters against the actual upstream dataset/model and simulator."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hydra,torch,numpy as np
from torch.utils.data import DataLoader
from square_upstream_adapters import config
ROOT=Path('/home/zxiao93/Documents/LaDiWM/results/dp_small100_20260923')
def main():
    torch.set_num_threads(4);checks={}
    for m in ['image','point_ae']:
        cfg,original=config(m,down_dims=[88,176,352])
        for key in ['optimizer','ema','dataloader','val_dataloader']:
            from omegaconf import OmegaConf
            assert OmegaConf.to_container(cfg[key],resolve=True)==original[key],key
        assert cfg.training.num_epochs==3050 and cfg.training.stop_after_epochs==100
        ds=hydra.utils.instantiate(cfg.task.dataset);vs=ds.get_validation_dataset()
        assert ds.train_mask.sum()==180 and vs.train_mask.sum()==20
        b=next(iter(DataLoader(ds,batch_size=2,num_workers=0)))
        p=hydra.utils.instantiate(cfg.policy);p.set_normalizer(ds.get_normalizer());p.cuda()
        b={k:({x:y.cuda() for x,y in v.items()} if isinstance(v,dict) else v.cuda()) for k,v in b.items()}
        loss=p.compute_loss(b);assert torch.isfinite(loss);loss.backward()
        assert p.action_dim==10
        p.eval();folder=ROOT/'preflight'/m;folder.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model={k:v.detach().cpu() for k,v in p.state_dict().items()},config=dict(modality=m,seed=42,epochs=100,smoke=False,down_dims=[88,176,352]),epoch=0),folder/'policy.pt')
        checks[m]=dict(train_samples=len(ds),validation_samples=len(vs),finite_loss=float(loss.detach()),action_dim=p.action_dim,trainable_parameters=sum(x.numel() for x in p.parameters() if x.requires_grad),parameter_entries_including_normalizer=sum(x.numel() for x in p.parameters()))
        del p,loss,b,ds,vs;torch.cuda.empty_cache()
        with (folder/'smoke.log').open('w') as f:
            subprocess.run([sys.executable,'scripts/eval_square_upstream.py','--run',str(folder),'--split','valid','--limit','1','--max-steps','2','--batch','1'],stdout=f,stderr=subprocess.STDOUT,check=True)
        result=json.loads((folder/'smoke/valid/results.json').read_text())
        assert result['batch1_upstream_sampler_parity'] and result['episodes'][0]['absolute_controller_verified'] and result['smoke']
        checks[m]['rollout_preflight_passed']=True
        (folder/'policy.pt').unlink() # Diagnostic random weights only, never a training checkpoint.
    (ROOT/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
if __name__=='__main__':main()
