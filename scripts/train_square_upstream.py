"""Run upstream workspace with an explicit 100-epoch early stop, not cosine100."""
import argparse, json, sys, time, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
from square_upstream_adapters import config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--modality',choices=['image','point_ae'],required=True);a=parser.parse_args()
    root=Path('/home/zxiao93/Documents/LaDiWM/results/dp_upstream_abs100_20260922')/f'{a.modality}_seed42'
    root.mkdir(parents=True,exist_ok=True)
    if (root/'complete.json').exists():return
    cfg,original=config(a.modality)
    assert cfg.training.num_epochs==3050 and cfg.training.stop_after_epochs==100
    OmegaConf.save(cfg,root/'config.yaml');(root/'original_config.json').write_text(json.dumps(original,indent=2)+'\n')
    (root/'code_sha256.json').write_text(json.dumps({str(p.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('square_upstream_adapters.py'),Path(__file__).with_name('eval_square_upstream.py'),Path(__file__).resolve().parents[1]/'diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py']},indent=2)+'\n')
    started=time.perf_counter();workspace=TrainDiffusionUnetHybridWorkspace(cfg,output_dir=str(root));workspace.run()
    if workspace._saving_thread:workspace._saving_thread.join()
    # Upstream periodic latest only reaches epoch 50 in a 100-epoch run.
    # Explicitly preserve the actual endpoint with model, EMA, and optimizer.
    workspace.save_checkpoint(path=root/'checkpoints/last_epoch_0099.ckpt',use_thread=False)
    endpoint=root/'last';endpoint.mkdir(exist_ok=True)
    torch.save(dict(model={k:v.detach().cpu() for k,v in workspace.ema_model.state_dict().items()},config=dict(modality=a.modality,seed=42,epochs=100,smoke=False),epoch=99),endpoint/'policy.pt')
    choices=[]
    for p in (root/'rollouts').glob('epoch_*/random_saved/results.json'):
        r=json.loads(p.read_text());choices.append((r['success_rate'],r['checkpoint_epoch'],str(p.parent.parent)))
    best=max(choices,key=lambda x:(x[0],-x[1]))
    (root/'complete.json').write_text(json.dumps(dict(epochs_completed=workspace.epoch,total_trainer_wall_seconds=time.perf_counter()-started,scheduler_epochs=3050,last_checkpoint=str(root/'checkpoints/last_epoch_0099.ckpt'),best_sr=best[0],best_epoch=best[1],best_rollout=best[2],selection='upstream periodic Random50 test/mean_score; earliest epoch breaks SR ties',python=sys.executable),indent=2)+'\n')
if __name__=='__main__':main()
