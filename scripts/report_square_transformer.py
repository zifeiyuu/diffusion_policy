"""Report measured51/100-epoch SR and costs for parameter-matched action diffusion."""
from pathlib import Path
import json,sys,shutil,re,statistics
L=Path('/home/zxiao93/Documents/LaDiWM');O=L/'results/dp_transformer100_20260923';R=L/'experiment_logs/2026-09-23/dp_transformer100'
sys.path.insert(0,str(L));from scripts.experiment_journal import register_report

def read(p):return json.loads(p.read_text()) if p.exists() else None
def mins(v):return 'Pending' if v is None else f'{v/60:.2f}'
def main():
 R.mkdir(parents=True,exist_ok=True);(R/'artifacts').mkdir(exist_ok=True)
 state=read(O/'queue_status.json') or {'state':'preparing'}
 details=[]
 rows=['| Model | Completed epochs | Eval20 SR | Random50 SR | Trainer wall min incl. periodic eval | Trainer wall min excl. periodic eval | Eval20 / Random50 wall min |','|---|---:|---:|---:|---:|---:|---:|']
 for m in ['image','point_ae']:
  root=O/f'{m}_seed42';complete=read(root/'complete.json');periodic=read(root/'rollouts/epoch_0050/rollout_wall.json')
  for epoch,sub in [(51,'rollouts/epoch_0050'),(100,'last')]:
   folder=root/sub;ev=[read(folder/s/'results.json') for s in ['valid','random_saved']]
   measured=read(folder/'training_time.json') if epoch==51 else complete
   wall=measured.get('trainer_wall_seconds_before_rollout') if measured and epoch==51 else measured.get('total_trainer_wall_seconds') if measured else None
   excluded=wall-(periodic['wall_seconds'] if epoch==100 and periodic else 0) if wall is not None else None
   sr=lambda e:f"{e['successes']}/{e['requested_episodes']} ({100*e['success_rate']:.1f}%)" if e and e['complete'] else 'Pending'
   rows.append(f"| {m} | {epoch} | {sr(ev[0])} | {sr(ev[1])} | {mins(wall)} | {mins(excluded)} | {' / '.join(mins(e['evaluator_wall_seconds']) if e and e['complete'] else 'Pending' for e in ev)} |")
   dest=R/m/str(epoch);dest.mkdir(parents=True,exist_ok=True)
   for name in ['training_time.json','rollout_wall.json','evaluation_provenance.json']:
    if (folder/name).exists():shutil.copy2(folder/name,dest/name)
   for split,e in zip(['valid','random_saved'],ev):
    if not e or not e['complete']:continue
    ms=e['task_makespan']['success_sim_seconds']
    fmt=lambda x:'NR' if x is None else f'{x:.2f}'
    details.append(f"| {m} | {epoch} | {split} | {fmt(ms['mean'])} / {fmt(ms['median'])} / {fmt(ms['p95'])} | {fmt(e['amortized_sampling_ms_per_episode_call'])} / {fmt(statistics.median(e['single_env_warm_sampling_ms']))} |")
    dst=dest/split;dst.mkdir(exist_ok=True)
    for p in (folder/split).iterdir():
     if p.suffix in ['.json','.npz','.mp4']:shutil.copy2(p,dst/p.name)
  dst=R/'artifacts'/m;dst.mkdir(exist_ok=True)
  for name in ['config.yaml','original_config.json','complete.json','code_sha256.json']:
   if (root/name).exists():shutil.copy2(root/name,dst/name)
 for name in ['parameter_audit.json','preflight.json']:
  if (O/name).exists():shutil.copy2(O/name,R/'artifacts'/name)
 body='''Question: evaluate upstream Transformer DP on image and frozen PointAE observations at51 and100 completed epochs, after the small UNet study.

Use the repository original train_diffusion_transformer_hybrid_workspace + square_image_abs configuration:8 layers,4 heads,embedding256,n_cond_layers0,causal attention,attention dropout0.3. Action diffusion Transformer has8,997,642 trainable parameters; image encoder22,394,176,total image31,391,818. PointAE action policy has8,997,642 trainable parameters plus frozen170-parameter encoder (full supplied AE2,540). No UNet enters these policies.

Seed42 for each modality;180/20 masks and saved Random50 resets unchanged. Preserve Transformer defaults: absolute10D actions,horizon10,observe2,execute8,batch64,DDPM100,EMA,AdamW learning_rate1e-4,betas[0.9,0.95],transformer weight_decay1e-3,observation encoder weight_decay1e-6,warmup1000,cosine3050-epoch budget. Stop after100 completed epochs and skip first-epoch rollout. Image encoder trains jointly; PointAE is frozen. At51 and100 epochs evaluate both Eval20 and Random50 and retain full periodic/final checkpoints. Fixed endpoints are reported separately, with no statistical-significance claim from a single seed. These are Transformer-specific defaults, not the UNet optimizer or horizon.


Policy/training use unchanged robodiff. Simulator uses dp_eval141 (robosuite1.4.1), restoring full saved XML and state. Absolute training labels reuse the previous upstream converter output generated in robodiff1.2; no data conversion change in this Transformer experiment. Dataset caches and AE preprocessing are reused. Previous full-size DP runs remain intact.

Cost:51-epoch training time is measured before its periodic rollout;100-epoch trainer wall includes that rollout, validation, setup and checkpointing. The excluded-rollout column subtracts the measured periodic-runner wall; it still includes offline validation and I/O. Final evaluations are separate. Per-result JSON records inference batch mean/P95, amortized latency, warm single-env latency, GPU allocation, full evaluator wall and success-only task makespan(steps/20Hz); failures censored at400 controls. Evaluation concurrency2. AE pretraining and data-cache construction are excluded and not newly measured; monetary/energy cost unrecorded.
'''
 body+='\nStatus: `'+str(state)+'`.\n\n'+'\n'.join(rows)+'\n\nLocal checkpoints/results: `'+str(O)+'`.\n'
 if details:body+='\n| Model | Epochs | Set | Success makespan s mean/median/P95 | Inference ms amortized / warm batch1 |\n|---|---:|---|---:|---:|\n'+'\n'.join(details)+'\n'
 (R/'README.md').write_text('# Original Transformer DP: image and PointAE,51/100 epochs\n\n'+body)
 register_report(R/'README.md',entry=body)
 p=L/'experiment_logs/summary/README.md';s=p.read_text();a='<!-- dp-transformer100:start -->';b='<!-- dp-transformer100:end -->'
 block=a+'\n## Original Transformer DP:51/100 epochs\n\n'+body+'\n[Evidence](../2026-09-23/dp_transformer100/README.md).\n'+b
 if a in s:s=re.sub(re.escape(a)+'.*?'+re.escape(b),lambda _:block,s,flags=re.S)
 else:
  title,rest=s.split('\n',1);s=title+'\n\n'+block+'\n'+rest
 p.write_text(s)
if __name__=='__main__':main()
