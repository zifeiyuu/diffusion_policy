"""Image Transformer baseline: seed42,batch128,exactly50 epochs; report in this repo."""
import json,os,sys,time,subprocess,shutil,re,fcntl
from pathlib import Path
D=Path(__file__).resolve().parents[1];sys.path.insert(0,str(D))
O=D/'data/outputs/square_transformer_b128_e50_seed42';R=D/'experiment_logs/square_transformer_b128_e50_seed42'
def read(p):return json.loads(p.read_text()) if p.exists() else None
def status(state,job):
 (O/'status.json').write_text(json.dumps(dict(state=state,job=job,updated=time.time()),indent=2)+'\n');report()
def report():
 R.mkdir(parents=True,exist_ok=True);t=read(O/'complete.json');ev=[read(O/'last'/s/'results.json') for s in ['valid','random_saved']]
 score=lambda e:f"{e['successes']}/{e['requested_episodes']} ({100*e['success_rate']:.1f}%)" if e and e['complete'] else 'Pending'
 timing=lambda e:f"{e['evaluator_wall_seconds']/60:.2f}" if e and e['complete'] else 'Pending'
 cost=f"{t['total_trainer_wall_seconds']/60:.2f}" if t else 'Pending'
 body='''## Square Image Transformer DP baseline:50 epochs,batch128

Fresh seed42 training with the upstream Transformer Hybrid Image policy:8 layers,4 heads,embedding256;8,997,642 action-network parameters plus22,394,176 image-encoder parameters (31,391,818 trainable total). Two84x84 RGB views and robot9 over two observation frames; image encoders train jointly from random initialization. Absolute10D policy actions,horizon10,execute8,DDPM100,EMA. Preserve Transformer AdamW defaults (LR1e-4,betas0.9/0.95,weight decay1e-3 Transformer /1e-6 image encoder),warmup1000,and cosine schedule over3050 epochs; stop at50 completed epochs. Only training batch changes to128; validation batch remains64. Train/validation masks180/20; evaluate exact Eval20 and saved Random50. No initial rollout or best-SR selection; evaluate final epoch49 EMA only. Training50 epochs at batch128 uses about half the updates of50 epochs at batch64.

Training/policy run in robodiff; simulation uses dp_eval141 (robosuite1.4.1) with full saved XML/state. Shared data and absolute-action cache are reused read-only from the prior study; action labels were generated with the upstream converter in robodiff1.2. Outputs and this report are stored in diffusion_policy, not LaDiWM. Checkpoint: `data/outputs/square_transformer_b128_e50_seed42/checkpoints/last_epoch_0049.ckpt` (model,EMA,optimizer); inference weights: `last/policy.pt` under the same output directory.

Trainer wall includes setup,offline validation,diagnostic sampling and checkpointing; no simulator rollout occurs during training. Final evaluator wall is separate and includes initialization,inference,simulation,rendering and saving. GPU allocation,batched and single-env inference timings,and success-only task makespan are recorded in result JSON. Failed episodes are censored at400 controls/20Hz. Reused AE/Stage1 costs are not applicable to this image model; no monetary/energy measurements.
'''
 body+='\nStatus: `'+str(read(O/'status.json'))+'`.\n\n| Train epochs | Batch | Eval20 SR | Random50 SR | Train wall min | Eval20 / Random50 wall min |\n|---:|---:|---:|---:|---:|---:|\n'+f'| 50 | 128 | {score(ev[0])} | {score(ev[1])} | {cost} | {timing(ev[0])} / {timing(ev[1])} |\n'
 for name in ['config.yaml','complete.json']:
  if (O/name).exists():shutil.copy2(O/name,R/name)
 for split,e in zip(['valid','random_saved'],ev):
  if e and e['complete']:
   shutil.copy2(O/'last'/split/'results.json',R/(split+'_results.json'))
   body+=f'\n[{split} measured results](experiment_logs/square_transformer_b128_e50_seed42/{split}_results.json).\n'
 (R/'README.md').write_text(body.replace('(experiment_logs/square_transformer_b128_e50_seed42/','('))
 p=D/'README.md';s=p.read_text();a='<!-- square-transformer-b128-e50:start -->';b='<!-- square-transformer-b128-e50:end -->';block=a+'\n'+body+'\n'+b
 if a in s:s=re.sub(re.escape(a)+'.*?'+re.escape(b),lambda _:block,s,flags=re.S)
 else:
  title,rest=s.split('\n',1);s=title+'\n\n'+block+'\n'+rest
 p.write_text(s)
def train():
 import torch
 from omegaconf import OmegaConf
 from square_upstream_adapters import config
 from diffusion_policy.workspace.train_diffusion_transformer_hybrid_workspace import TrainDiffusionTransformerHybridWorkspace
 if (O/'complete.json').exists():return
 cfg,_=config('image',architecture='transformer');cfg.dataloader.batch_size=128;cfg.training.stop_after_epochs=50
 assert cfg.training.num_epochs==3050 and cfg.training.skip_initial_rollout
 OmegaConf.save(cfg,O/'config.yaml');status('running','training50 epochs,batch128')
 start=time.perf_counter();w=TrainDiffusionTransformerHybridWorkspace(cfg,output_dir=str(O));w.run()
 if w._saving_thread:w._saving_thread.join()
 w.save_checkpoint(path=O/'checkpoints/last_epoch_0049.ckpt',use_thread=False)
 (O/'last').mkdir(exist_ok=True)
 torch.save(dict(model={k:v.detach().cpu() for k,v in w.ema_model.state_dict().items()},config=dict(modality='image',seed=42,epochs=50,smoke=False,architecture='transformer',down_dims=None),epoch=49),O/'last/policy.pt')
 (O/'complete.json').write_text(json.dumps(dict(epochs_completed=w.epoch,total_trainer_wall_seconds=time.perf_counter()-start,scheduler_epochs=3050,batch_size=128,global_step=w.global_step,python=sys.executable),indent=2)+'\n')
def main():
 os.chdir(D);O.mkdir(parents=True,exist_ok=True)
 lock=(O/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline',PYTHONUNBUFFERED='1')
 try:
  status('waiting','Waiting for existing GPU compute jobs to finish')
  while True:
   active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if not active:break
   time.sleep(30)
  train()
  import torch
  torch.cuda.empty_cache()
  for split in ['valid','random_saved']:
   status('running','final '+split)
   with (O/(split+'.log')).open('a') as f:subprocess.run([sys.executable,'scripts/eval_square_upstream.py','--run',str(O/'last'),'--split',split],stdout=f,stderr=subprocess.STDOUT,check=True)
   report()
  status('complete','50 epochs and both evaluation sets complete')
 except BaseException as e:status('failed',str(e));raise
if __name__=='__main__':main()
