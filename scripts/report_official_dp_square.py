"""Build the user-selected official DP report from six completed evaluations."""
import json
from pathlib import Path
import shutil
import statistics
import sys
ROOT=Path('/home/zxiao93/Documents/LaDiWM')
sys.path.insert(0,str(ROOT))
from scripts.experiment_journal import register_report
P=ROOT/'experiment_logs/2026-09-22/official_dp_square'
RUN=ROOT/'results/official_dp_square_20260922'

def fmt(x):return 'NR' if x is None else f'{x:.2f}'
def build():
    results={(i,s):json.loads((P/f'train_{i}'/s/'results.json').read_text()) for i in range(3) for s in ['valid','random_saved']}
    assert all(r['complete'] and r['completed_episodes']==r['requested_episodes'] and r['batch1_upstream_sampler_parity'] for r in results.values())
    provenance=[json.loads((P/'artifacts'/f'train_{i}_provenance.json').read_text()) for i in range(3)]
    for s in ['valid','random_saved']:
        key='validation_demo_names' if s=='valid' else 'initial_bank_sha256'
        assert all(results[i,s][key]==results[0,s][key] for i in range(3))
    lines=['| Published run / seed / epoch | Eval20 SR | Random50 SR | Original train / new train min | Eval wall min: 20 / 50 | Random50 sampling ms: batch-amortized / warm single-env median | Peak allocated GPU MiB: 20 / 50 |', '|---|---:|---:|---|---:|---:|---:|']
    for i,p in enumerate(provenance):
        v,r=results[i,'valid'],results[i,'random_saved']
        lines.append(f"| train_{i} / {p['training_seed']} / {p['epoch']} | {v['successes']}/20 ({100*v['success_rate']:.0f}%) | {r['successes']}/50 ({100*r['success_rate']:.0f}%) | External NR / 0 | {v['evaluator_wall_seconds']/60:.2f} / {r['evaluator_wall_seconds']/60:.2f} | {r['amortized_sampling_ms_per_episode_call']:.2f} / {statistics.median(r['single_env_warm_sampling_ms']):.2f} | {v['peak_cuda_allocated_bytes']/2**20:.1f} / {r['peak_cuda_allocated_bytes']/2**20:.1f} |")
    means={s:(statistics.mean([100*results[i,s]['success_rate'] for i in range(3)]),statistics.stdev([100*results[i,s]['success_rate'] for i in range(3)])) for s in ['valid','random_saved']}
    conclusion=f"Across three published training seeds, SR is **{means['valid'][0]:.2f}% ± {means['valid'][1]:.2f} pp on Eval20** and **{means['random_saved'][0]:.2f}% ± {means['random_saved'][1]:.2f} pp on Random50** (mean ± sample SD). These are paired initial-state evaluations of externally trained, publisher-selected checkpoints; they do not isolate architecture or representation effects against E39."
    registry={e['id']:e for e in json.loads((ROOT/'experiment_logs/experiments.json').read_text())}
    baseline=[registry[k] for k in ['E98','E79','E81']]
    compare=['| Training seed | No-future ID | DP Eval20 / Random50 | KUBM + diffusion, no future Eval20 / Random50 | DP minus no future: Eval20 / Random50 (pp) |', '|---|---|---:|---:|---:|']
    baseline_rates=[]
    for i,e in enumerate(baseline):
        v=e['eval20_successes']/20*100;r=e['random50_successes']/50*100
        dv=results[i,'valid']['success_rate']*100;dr=results[i,'random_saved']['success_rate']*100
        baseline_rates.append((v,r))
        compare.append(f"| {provenance[i]['training_seed']} | {e['id']} | {dv:.0f}% / {dr:.0f}% | {v:.0f}% / {r:.0f}% | {dv-v:+.2f} / {dr-r:+.2f} |")
    bv=statistics.mean(x[0] for x in baseline_rates);br=statistics.mean(x[1] for x in baseline_rates)
    compare.append(f"| Three-seed mean | E98/E79/E81 | {means['valid'][0]:.2f}% / {means['random_saved'][0]:.2f}% | {bv:.2f}% / {br:.2f}% | **{means['valid'][0]-bv:+.2f} / {means['random_saved'][0]-br:+.2f}** |")
    conclusion += '\n\n**User-selected comparison: KUBM + diffusion with no future, three seeds.**\n\n'+'\n'.join(compare)
    conclusion += '\n\n[No-future source report (E98/E79/E81)](https://github.com/zifeiyuu/koopman_diffusion_ladiwm/blob/main/experiment_logs/2026-09-21/nofuture_three_seeds/README.md): seeds42/43/44,100 Stage2 epochs, final epoch99, frozen KUBM history encoder and PointAE, no future tokens. Historical51-epoch E20 is excluded. No-future reused Stage2 training wall is4.73/4.67/4.68 min; full evaluator wall (Eval20/Random50) is1.04/2.26,0.95/2.16,0.94/2.15 min. Shared Stage1/AE costs are not charged again. No new training or evaluation is performed for this comparison; DP costs follow below. These are descriptive percentage-point gaps, not an equal-budget or same-simulator ablation: DP uses publisher-selected epochs1250/2050/2600 and native Robosuite1.2.0, whereas no-future uses100 epochs and Robosuite1.4.1. Inputs, splits, actions, samplers and evaluation concurrency also differ. Three seeds do not establish statistical significance.'
    table='\n'.join(lines)
    costs='Measured on one RTX 4080 SUPER for policy inference, up to eight concurrent CPU-rendered official simulator environments, original DDPM100. Full evaluator wall includes checkpoint loading, reset/startup, sampler verification/warmup, episodes, artifact writing and worker shutdown. Sampling covers the image encoder, DDPM100 and output transfer, synchronized with CUDA. Batch-amortized sampling time is batch wall divided by episode-policy calls, not single-environment latency; the separate warm single-environment figure is the median of three measured calls. Peak GPU is PyTorch allocation, excluding simulator/driver allocations. Original external training wall/energy/money cost is unavailable; no new training was performed. Historical single-environment evaluator walls are not throughput-matched to this batched run. Environment setup/downloads, the standalone70-reset audit and the cancelled1.4.1 attempt are outside these six evaluator walls; their combined setup/aborted-job wall was not measured.'
    costs += f" The six formal evaluator walls sum to {sum(r['evaluator_wall_seconds'] for r in results.values())/60:.2f} minutes."
    duration=['| Run | Set | Successes | Successful task makespan seconds: mean / median / P95 | Mean success steps | Failures / mean censored steps | Sum overlapping episode-loop min | Summed policy-call wall min |','|---|---|---:|---:|---:|---:|---:|---:|']
    for i in range(3):
        for s in ['valid','random_saved']:
            r=results[i,s];t=r['task_makespan'];d=t['success_sim_seconds']
            duration.append(f"| train_{i} | {'Eval20' if s=='valid' else 'Random50'} | {t['success_count']} | {fmt(d['mean'])} / {fmt(d['median'])} / {fmt(d['p95'])} | {fmt(t['success_steps']['mean'])} | {t['failure_count']} / {fmt(t['failure_steps']['mean'])} | {r['episode_loop_seconds']/60:.2f} | {r['sampling_total_seconds']/60:.2f} |")
    duration='\n'.join(duration)
    intro='**Question:** how do the three officially released Square PH image CNN Diffusion Policy checkpoints perform on the LaDiWM Eval20 and saved Random50 state vectors in the official native simulator? **Settings:** published score-selected EMA checkpoints, two RGB84 cameras and measured robot pose/gripper qpos, two observation frames, predict16/execute8, DDPM100, absolute OSC actions, at most400 control steps at20Hz, stop on first task success.'
    limits='The official saved config trains on Square PH `image_abs.hdf5`, with a random 2% validation split (dataset seed42), not the LaDiWM180/20 mask. Therefore Eval20 is reset-aligned and must not be called held out from these checkpoints. The saved configured maximum is8000 epochs; the evaluated saved epochs are1250/2050/2600, selected by the publisher before this evaluation, not3050-epoch retraining or latest checkpoints. E39 uses privileged projected points, a different action representation, history, architecture and sampler; SR differences do not establish a controlled image-versus-flow comparison. Random50 is the existing reused same-task placement bank. Policy and simulator both run in the diffusion_policy robodiff environment, with official cheng-chi/robosuite commit277ab958 (version1.2.0), free-mujoco-py2.1.6 and robomimic0.2.0. The original LaDiWM1.4.1 model XML is incompatible with old MuJoCo, so the saved state vectors are restored into the native official scene after checking joint order. Table, peg, robot-base and camera poses and all70 state restores were audited. These are state-paired evaluations, not identical scene XML, pixels or physics to LaDiWM. This is also not the original publisher reset bank. The earlier accidental kguide/1.4.1 evaluation was stopped and excluded; its partial results remain only in ignored results/official_dp_square_20260922/nonofficial_robosuite141.'
    encoder='Both camera ResNet18 + SpatialSoftmax encoders were initialized without pretrained image weights and optimized jointly with the diffusion UNet. The official optimizer uses `self.model.parameters()` and robomimic image config sets `backbone_kwargs.pretrained=False`. The checkpoint EMA state includes264 `obs_encoder.*` tensors; strict loading restores the22,394,176 encoder parameters together with255,612,042 diffusion parameters and observation/action normalization. Evaluation freezes the loaded policy; it does not train an encoder or an AE.'
    sources=json.loads((RUN/'downloads.json').read_text())
    links='\n'.join(f"- [train_{i}, published checkpoint]({sources[i]['url']}): SHA256 `{provenance[i]['checkpoint_sha256']}`; [resolved config and provenance](artifacts/train_{i}_provenance.json)." for i in range(3))
    evidence='\n'.join(f"- train_{i} {s}: [raw results](train_{i}/{s}/results.json); [video0](train_{i}/{s}/episode_000.mp4), [video1](train_{i}/{s}/episode_001.mp4), [video2](train_{i}/{s}/episode_002.mp4). Initial-state and action-trace NPZ files are alongside these videos." for i in range(3) for s in ['valid','random_saved'])
    protocol='Eval20 follows `source/low_dim_v141.hdf5` mask/valid order. Random50 uses `results/kubm_square_all200/eval/20260916_230541/initial_000.npz` through049. Every restored MuJoCo state is checked at absolute tolerance1e-10, with the source and native scene joint order asserted equal; Random50 source SHA256s and Eval20 demo names are in each results JSON. Each episode has an independent CUDA RNG seeded42+episode index, shared across models. Observations are updated every executed control step. The original scheduler.step is retained; batch1 sampling is checked bit-for-bit against upstream for every evaluator. The published10D absolute action is decoded with upstream RotationTransformer (rotation6D to axis-angle); official robosuite1.2.0 uses control_delta=False and each submitted XYZ is checked against controller.goal_pos. RGB frames are vertically flipped simulator renders,84×84. Task makespan is executed steps to first success divided by20; failed episodes are censored and excluded from success-duration statistics. Overlapping episode wall sums are not total elapsed compute.'
    report=P/'README.md'
    report.write_text('# Official image CNN Diffusion Policy — three published seeds\n\n'+intro+'\n\n'+conclusion+'\n\n'+table+'\n\n'+costs+'\n\n'+duration+'\n\n'+protocol+'\n\n'+limits+'\n\n'+encoder+'\n\n## Checkpoint provenance\n\n'+links+'\n\n[Published config](artifacts/published_config.yaml); [download manifest](artifacts/downloads.json); [official pinned simulator version](https://github.com/cheng-chi/robosuite/blob/277ab9588ad7a4f4b55cf75508b44aa67ec171f0/robosuite/__init__.py). Model binaries remain under ignored `results/official_dp_square_20260922/checkpoints/` (about13GiB combined), not Git.\n\n## Environment and reset audit\n\n[Final210-episode integrity check](artifacts/final_integrity_check.json), [All70 state and fixed-scene-pose checks](artifacts/native_official_reset_audit.json), [runtime versions and executable](artifacts/runtime_versions.json), [installed Python packages](artifacts/robodiff_pip_freeze.txt), [official environment YAML](artifacts/official_conda_environment.yaml). MuJoCo-py used its CPU renderer; policy inference used CUDA. The official simulator wheel declares an old numba dependency conflicting with the YAML; numba0.56.4 from the YAML was restored after installation.\n\n## Evidence and videos\n\n'+evidence+'\n\n## Reproduce\n\nUse the [diffusion_policy evaluator release](https://github.com/zifeiyuu/diffusion_policy/tree/4e7e968), robodiff for both policy inference and the official robosuite1.2.0 simulator, from the diffusion_policy checkout. Set `DP_REPO_ROOT` if that checkout is elsewhere; dataset and environment paths are defined in the evaluator. Download the three URLs above to the paths in the manifest. Run each of the six combinations:\n\n```bash\n/home/zxiao93/anaconda3/envs/robodiff/bin/python scripts/eval_official_dp_square.py --run 0 --split valid --batch 8\n# --run 0/1/2; --split valid/random_saved\npython scripts/report_official_dp_square.py\n```\n')
    for name in ['downloads.json','published_config.yaml']:shutil.copy2(RUN/name,P/'artifacts'/name)
    entry=intro+'\n\n'+conclusion+'\n\n'+table+'\n\n'+costs+'\n\n'+limits
    register_report(report,entry=entry)
    summary=ROOT/'experiment_logs/summary/README.md';text=summary.read_text()
    start='<!-- official-dp-square:start -->';end='<!-- official-dp-square:end -->'
    block=start+'\n## Official image DP: three published seeds (2026-09-22, user-selected)\n\n'+intro+'\n\n'+conclusion+'\n\n'+table+'\n\n'+duration+'\n\n'+costs+'\n\n'+limits+'\n\n**Image encoder:** jointly trained from random initialization; EMA checkpoint includes both image encoders and the diffusion network. [Full report, exact checkpoints, protocol, raw results and videos](../2026-09-22/official_dp_square/README.md).\n'+end
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+block+text[b:]
    else:
        title,rest=text.split('\n',1);text=title+'\n\n'+block+'\n'+rest
    summary.write_text(text)
    print(conclusion)
if __name__=='__main__':build()
