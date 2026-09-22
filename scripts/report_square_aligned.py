"""Maintain the user-requested Summary block plus the LaDiWM experiment journal."""
import os
import re
from square_dp_data import *
from square_dp_metrics import task_metrics


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def render():
    REPORT.mkdir(parents=True,exist_ok=True)
    status=read(OUT/'queue_status.json') or dict(state='preparing')
    rows=[];duration_rows=[];done=[];completed=[]
    for modality in ['image','point_flow']:
        for seed in [42,43,44]:
            folder=OUT/f'{modality}_seed{seed}'
            training=read(folder/'complete.json');progress=read(folder/'progress.json')
            evaluations=[read(folder/s/'results.json') for s in ['valid','random_saved']]
            scores=[f"{e['successes']}/{e['requested_episodes']} ({100*e['success_rate']:.1f}%)" if e and e['complete'] else 'Pending' for e in evaluations]
            epoch=f"{progress['epoch']+1}/3050" if progress else 'Queued'
            train_cost=f"{training['total_trainer_wall_seconds']/60:.2f}" if training else (f"{progress['train_wall_seconds']/60:.2f} so far" if progress else 'NR')
            ev_cost=' / '.join(f"{e['evaluator_wall_seconds']/60:.2f}" if e and e['complete'] else 'NR' for e in evaluations)
            peak=f"{progress['peak_cuda_allocated_bytes']/2**20:.0f}" if progress else 'NR'
            peak_eval=f"{evaluations[1]['peak_cuda_allocated_bytes']/2**20:.0f}" if evaluations[1] and evaluations[1]['complete'] else 'NR'
            latency=f"{evaluations[1]['sampling_mean_ms']:.2f}" if evaluations[1] and evaluations[1]['complete'] else 'NR'
            label=f"DP-{'RGB' if modality=='image' else 'FLOW'}-{seed}"
            for split,e in zip(['Eval20','Random50'],evaluations):
                if e and e['complete']:
                    m=e.get('task_makespan') or task_metrics(e['episodes'])
                    def fmt(v):return 'N/A' if v is None else f'{v:.2f}'
                    duration_rows.append(f"| {label} | {split} | {m['success_count']}/{e['requested_episodes']} | {fmt(m['success_steps']['mean'])} | {fmt(m['success_sim_seconds']['mean'])} / {fmt(m['success_sim_seconds']['median'])} / {fmt(m['success_sim_seconds']['p95'])} | {fmt(m['success_execution_wall_seconds']['mean'])} | {m['failure_count']} / {fmt(m['failure_steps']['mean'])} | {e['sampling_mean_ms']:.2f} / {e['sampling_p95_ms']:.2f} |")
                else:
                    duration_rows.append(f'| {label} | {split} | Pending | NR | NR | NR | NR | NR |')

            rows.append(f'| {label} | {epoch} | {scores[0]} | {scores[1]} | {train_cost} | {ev_cost} | {peak} / {peak_eval} | {latency} |')
            if training and all(e and e['complete'] for e in evaluations):
                completed.append(dict(modality=modality,seed=seed,train=training,eval=evaluations))
            done.append(dict(modality=modality,seed=seed,training=training,progress=progress,evaluations=evaluations))
    table='\n'.join(['| Run | Epochs | Eval20 SR | Random50 SR | Train wall min | Eval wall min: 20 / 50 | Peak GPU MiB: train / eval | Sample ms/call |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']+rows)
    table+='\n\n'+ '\n'.join(['| Run | Set | Successes | Success makespan: mean steps | Success makespan seconds: mean / median / P95 | Success execution wall: mean seconds | Failures / mean censored steps | Inference ms/call: mean / P95 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']+duration_rows)
    aggregates=[]
    for modality in ['image','point_flow']:
        runs=[r for r in completed if r['modality']==modality]
        if len(runs)==3:
            vals=[]
            for i in [0,1]:
                rates=np.array([r['eval'][i]['success_rate']*100 for r in runs])
                vals.append(f'{rates.mean():.1f}% ± {rates.std(ddof=1):.1f} pp')
            aggregates.append(f"**{modality}, three seeds:** Eval20 {vals[0]}; Random50 {vals[1]} (mean ± sample SD).")
            def aggregate(values):
                values=[x for x in values if x is not None]
                return f'{np.mean(values):.2f} ± {np.std(values,ddof=1):.2f}' if len(values)==3 else 'N/A (some seeds have no successes)'
            aggregates.append(f"**{modality} cost across seeds:** training wall min {aggregate([r['train']['total_trainer_wall_seconds']/60 for r in runs])}.")
            for i,label in enumerate(['Eval20','Random50']):
                es=[r['eval'][i] for r in runs]
                ms=[e.get('task_makespan') or task_metrics(e['episodes']) for e in es]
                aggregates.append(f"{modality} {label}, mean ± sample SD over seeds: evaluator wall min {aggregate([e['evaluator_wall_seconds']/60 for e in es])}; inference ms/call {aggregate([e['sampling_mean_ms'] for e in es])}; success makespan seconds {aggregate([m['success_sim_seconds']['mean'] for m in ms])}. Each makespan is conditional on success and must be read together with SR.")

    conclusion=('All six runs and 420 paired evaluation episodes are complete. Results are descriptive across three training seeds.'
        if len(completed)==6 else f'{len(completed)}/6 runs have finished training and both evaluation sets. No final comparative conclusion yet.')
    question='Question: how does retrained image Diffusion Policy compare with the same action diffusion network conditioned on causal 2D surface-point coordinates and displacement?'
    protocol='''Both encoders start randomly and train jointly with action denoising; no pretrained image weights, PointAE, Koopman model, or auxiliary AE pretraining. Seeds42/43/44, 3050 epochs each. Upstream `DiffusionUnetHybridImagePolicy`, UNet widths512/1024/2048, embedding128, kernel5, GroupNorm; DDPM100 cosine epsilon prediction at training and evaluation, EMA power0.75. AdamW1e-4, betas0.95/0.999, weight decay1e-6, batch64, 500-update warmup then cosine. Evaluate the **final epoch3049 EMA**; no best-SR selection or online rollout selection during training.

Both variants use two observation frames, prediction horizon16 and execution8 (action slots1–8), 7D delta OSC actions, and the same measured EE position/quaternion/gripper qpos. Image: independent randomly initialized ResNet18 + SpatialSoftmax encoders for agentview and wrist RGB84, random crop76 during training / center crop at evaluation, 64D each. Point-flow: 256 fixed corresponding Square surface points, including occluded points, projected in agentview; MLP1024→256→128 with ReLU. Input is normalized XY plus backward displacement. To match RGB's two-frame temporal access, the older token's displacement is zero; the current token contains p[t]−p[t−1], zero at t0. Visual embeddings are128D for both; both condition the same action UNet on137D/frame including robot9. The point observation is privileged simulator geometry and has no wrist view. Encoder architectures and parameter counts differ.

Exact HDF5 mask/train180 and mask/valid20 are reused across seeds; all normalizer statistics use only train180. Original DP sequence sampling (horizon16, pad_before1, pad_after7, repeated endpoint actions) is retained, giving 25,905 training / 2,849 validation windows; this differs from LaDiWM's every-frame213-updates/epoch training and 7/7 prediction/execution. This is a split/reset-aligned DP comparison, not an equal-compute or identical-input reproduction of E39. Eval20 restores the same validation-demo states; Random50 restores the original saved bank. DDPM terminal floating-point overshoot is clipped to controller input bounds[-1,1] (maximum recorded in JSON; values>1e-3 are rejected). Restore errors must be≤1e-10; seed42+i after each reset;400 steps maximum, stop on success,20Hz. Unlike original LaDiWM Stage1, neither encoder trains on the20 validation demos. Validation loss is logged only, never used to select the reported checkpoint.

Simulator runs in existing `kguide` with robosuite1.4.1; DP trains in `robodiff` with torch1.12.1. Missing robomimic0.2.0 was installed and huggingface-hub pinned to0.25.2 for diffusers0.11.1 compatibility. NumPy1/2 IPC uses explicit array bytes rather than cross-version NumPy pickle.'''
    costs='Task makespan: steps to first success /20Hz; report success-only mean, median and P95 alongside SR. Failed episodes are censored and reported separately, never counted as successful completion times; no successes means N/A. Execution wall includes inference, simulator, observation preparation and in-loop video recording, excluding reset/startup/final artifact writes. Inference timing covers the encoder plus all100 denoising steps and action output transfer, with CUDA synchronization. Costs: measured on one RTX4080SUPER. Training wall includes trainer setup, validation and checkpointing; summed epoch compute is retained separately in JSON. Evaluator wall includes model/simulator startup and videos; episode-loop time is separate. GPU is peak PyTorch allocation, not driver memory. No checkpoint/training reuse across the six runs. No monetary or energy cost measured. NR means not recorded yet. Smoke tests are separate and excluded from SR aggregates.'
    details=[]
    for r in completed:
        folder=OUT/f"{r['modality']}_seed{r['seed']}"
        rel=os.path.relpath(folder,REPORT)
        details.append(f"- {r['modality']} seed{r['seed']}: [training]({rel}/complete.json), [Eval20]({rel}/valid/results.json), [Random50]({rel}/random_saved/results.json), [Eval20 video]({rel}/valid/episode_000.mp4), [Random50 video]({rel}/random_saved/episode_000.mp4). Summed epoch compute: {r['train']['epoch_sum_seconds']/60:.2f} min; episode-loop20/50: {r['eval'][0]['episode_loop_seconds']/60:.2f}/{r['eval'][1]['episode_loop_seconds']/60:.2f} min.")
    body=f'# Retrained image DP versus causal 2D point-flow DP: 3050 epochs, three seeds\n\n{question}\n\nQueue: **{status["state"]}**; current job: `{status.get("job","none")}`.\n\n{table}\n\n'+ '\n\n'.join(aggregates)+f'\n\n{conclusion}\n\n## Protocol\n\n{protocol}\n\n## Cost and evidence\n\n{costs}\n\n[Data / simulator audit](artifacts/audit.json).\n\n'+'\n'.join(details)+f'\n\n## Reproduce / resume\n\n`bash /home/zxiao93/Documents/diffusion_policy/scripts/run_square_aligned.sh`\n\nArtifacts and logs: `{OUT}`. The queue resumes saved epoch checkpoints; a lock prevents duplicate launches. Checkpoint snapshots are saved every10 epochs; final EMA checkpoints are retained. Per-run resumable optimizer checkpoints are removed only after both full evaluations have succeeded.\n'
    (REPORT/'README.md').write_text(body)
    write_json(REPORT/'artifacts/summary.json',dict(queue=status,runs=done,completed_runs=len(completed)))
    sys.path.insert(0,str(LADIWM))
    from scripts.experiment_journal import register_report
    entry=question+'\n\n'+table+'\n\n'+conclusion+'\n\n'+costs
    register_report(REPORT/'README.md',entry=entry)
    # Explicitly requested by the user; preserve all A–I groups and historical IDs.
    summary=LADIWM/'experiment_logs/summary/README.md'
    text=summary.read_text();begin='<!-- dp-square-3050:start -->';end='<!-- dp-square-3050:end -->'
    rel=os.path.relpath(REPORT/'README.md',summary.parent)
    block=f'{begin}\n### Retrained DP: image versus causal 2D point flow (user-selected)\n\nRandomly initialized encoders trained jointly with DP; no pretrained PointAE. Three seeds42/43/44,3050 epochs each, final EMA, same180/20 split and Eval20/Random50 resets. Two observations,predict16/execute8,DDPM100. Point-flow observes privileged projected geometry. [Full settings, progress, costs and evidence]({rel}).\n\n{table}\n\n'+ '\n\n'.join(aggregates)+f'\n\n{conclusion}\n\n{costs}\n{end}'
    if begin in text:
        text=re.sub(re.escape(begin)+'.*?'+re.escape(end),lambda _:block,text,flags=re.S)
    else:
        assert '\n## B. Future length' in text
        text=text.replace('\n## B. Future length','\n'+block+'\n\n## B. Future length',1)
    tmp=summary.with_suffix('.tmp.md');tmp.write_text(text);tmp.replace(summary)


if __name__=='__main__':render()
