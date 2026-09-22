"""Report one-seed100-epoch RGB versus frozen PointAE DP without altering old runs."""
from pathlib import Path
import json,os,sys,re,shutil
L=Path('/home/zxiao93/Documents/LaDiWM');O=L/'results/dp_image_pointae_100_20260922';R=L/'experiment_logs/2026-09-22/dp_image_pointae_100'
sys.path.insert(0,str(L))
from scripts.experiment_journal import register_report

def read(p):return json.loads(p.read_text()) if p.exists() else None
def fmt(v):return 'NR' if v is None else f'{v:.2f}'
def main():
    R.mkdir(parents=True,exist_ok=True);state=read(O/'queue_status.json') or {'state':'running','job':'image_train'}
    rows=['| Modality, seed42 | Epochs | Eval20 SR | Random50 SR | Train wall min | Eval wall min:20/50 |','|---|---:|---:|---:|---:|---:|'];duration=[];finished=0
    for m in ['image','point_ae']:
        run=O/f'{m}_seed42';p=read(run/'progress.json');t=read(run/'complete.json');ev=[read(run/s/'results.json') for s in ['valid','random_saved']]
        scores=[f"{e['successes']}/{e['requested_episodes']} ({100*e['success_rate']:.1f}%)" if e and e['complete'] else 'Pending' for e in ev]
        rows.append(f"| {m} | {str(p['epoch']+1)+'/100' if p else 'Queued'} | {' | '.join(scores)} | {fmt(t['total_trainer_wall_seconds']/60) if t else (fmt(p['train_wall_seconds']/60)+' so far' if p else 'NR')} | {' / '.join(fmt(e['evaluator_wall_seconds']/60) if e and e['complete'] else 'NR' for e in ev)} |")
        if t and all(e and e['complete'] for e in ev):finished+=1
        for split,e in zip(['valid','random_saved'],ev):
            if not e or not e['complete']:continue
            a=e['task_makespan']['success_sim_seconds'];duration.append(f"| {m} | {split} | {a['count']} | {fmt(a['mean'])} / {fmt(a['median'])} / {fmt(a['p95'])} | {fmt(e['sampling_mean_ms'])} / {fmt(e['sampling_p95_ms'])} | {fmt(e.get('frozen_ae_total_seconds'))} | {fmt(e['peak_cuda_allocated_bytes']/2**20)} |")
            dest=R/m/split;dest.mkdir(parents=True,exist_ok=True)
            for path in (run/split).iterdir():
                if path.suffix in ['.json','.npz','.mp4']:shutil.copy2(path,dest/path.name)
        dest=R/'artifacts'/m;dest.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','progress.json','complete.json','epochs.jsonl']:
            if (run/name).exists():shutil.copy2(run/name,dest/name)
    intro='Question: at100 epochs and seed42, how does RGB DP compare with the same DP backbone conditioned on frozen PointAE features? Both train from scratch on the same180/20 mask, batch64,405 updates/epoch,40,500 updates total, final epoch99 EMA, AdamW1e-4, cosine schedule over100 epochs, DDPM100. Two observed frames, predict16/execute8,7D delta OSC actions. RGB: two84x84 cameras with jointly trained random-init ResNet18 encoders. PointAE: LaDiWM pretrained frozen170-parameter encoder of current/history256 projected points in raw pixel XY;128D feature/frame plus identical robot9. This is the cached-AE feature variant, not the raw XY/displacement MLP variant; no KUBM model or future rollout enters either DP.'
    limits='Policy and evaluation simulator run in robodiff with official Robosuite1.2.0, native scene, delta controller. Training RGB and point tracks come from the existing v1.4.1 replay dataset; evaluation therefore has a simulator/asset version difference. Eval20 and saved Random50 state vectors are restored with joint-order/state checks; original1.4.1 XML is not loaded. Point geometry is privileged, includes occluded points, and has no wrist camera. AE pretraining is reused and its original cost/data exposure is not attributed to fresh DP training. This single-seed, fixed-epoch experiment does not match encoder pretraining, model size, or full compute cost. It also differs from the published absolute-action DP checkpoints.'
    costs='Train wall includes setup, validation and checkpointing; summed training-epoch compute remains in JSON. Eval wall includes model/env startup, reset, sampling, simulation, videos and artifact saving. Sampling mean/P95 includes DP encoder/denoising/output transfer; AE forward cost is separately measured every observation and included in combined_policy_compute_seconds. Success makespan is steps/20Hz; failed episodes are censored at400 steps. Peak GPU is PyTorch allocation. Original AE pretraining cost is unrecorded; no monetary/energy measurement. Cached data/features are reused; cache preparation, if performed, is distinct from per-run training.'
    evidence='\n'.join(f'- {m} {s}: [results]({m}/{s}/results.json), [video0]({m}/{s}/episode_000.mp4), [video1]({m}/{s}/episode_001.mp4), [video2]({m}/{s}/episode_002.mp4).' for m in ['image','point_ae'] for s in ['valid','random_saved'] if (R/m/s/'results.json').exists())
    table='\n'.join(rows);dtable=('\n\n| Modality | Set | Successes | Success makespan sec: mean/median/P95 | DP sampling ms:mean/P95 | Total AE forward sec | Peak GPU MiB |\n|---|---|---:|---:|---:|---:|---:|\n'+'\n'.join(duration)) if duration else ''
    conclusion=f"Status: **{state['state']}**, job `{state.get('job','')}`. {finished}/2 models have completed training and both evaluation sets. Results are final only when both models finish; no comparative conclusion from partial results."
    body=intro+'\n\n'+conclusion+'\n\n'+table+dtable+'\n\n'+costs+'\n\n'+limits
    (R/'README.md').write_text('# Image DP versus frozen PointAE DP:100 epochs,seed42\n\n'+body+'\n\n## Evidence\n\n'+evidence+'\n\nRun from diffusion_policy with `robodiff/bin/python scripts/queue_square_100.py`. Local outputs: `'+str(O)+'`. Original3050-epoch runs remain stopped and separate.\n')
    register_report(R/'README.md',entry=body)
    summary=L/'experiment_logs/summary/README.md';text=summary.read_text();a='<!-- dp-square-100-ae:start -->';b='<!-- dp-square-100-ae:end -->'
    block=a+'\n## Image versus frozen PointAE DP:100 epochs,seed42 (user-selected)\n\n'+body+'\n\n[Full evidence and videos](../2026-09-22/dp_image_pointae_100/README.md).\n'+b
    if a in text:text=re.sub(re.escape(a)+'.*?'+re.escape(b),lambda _:block,text,flags=re.S)
    else:
        title,rest=text.split('\n',1);text=title+'\n\n'+block+'\n'+rest
    summary.write_text(text)
if __name__=='__main__':main()
