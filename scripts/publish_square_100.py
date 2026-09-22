"""Validate complete results and publish only this authorized experiment's reports."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
L=Path('/home/zxiao93/Documents/LaDiWM');O=L/'results/dp_image_pointae_100_20260922';R=L/'experiment_logs/2026-09-22/dp_image_pointae_100'
def outside(name):
    text=(L/name).read_text()
    if '/summary/' in name:a,b='<!-- dp-square-100-ae:start -->','<!-- dp-square-100-ae:end -->'
    else:
        key='experiment_logs/2026-09-22/dp_image_pointae_100/README.md';a=f'<!-- experiment:{key}:start -->';b=f'<!-- experiment:{key}:end -->'
    text=re.sub(re.escape(a)+'.*?'+re.escape(b),'',text,flags=re.S)
    return hashlib.sha256(text.encode()).hexdigest()
def main():
    pages=['experiment_logs/summary/README.md','experiment_logs/others/README.md'];guard=O/'publish_guard.json'
    if '--init' in sys.argv:
        guard.write_text(json.dumps({p:outside(p) for p in pages},indent=2)+'\n');return
    expected=json.loads(guard.read_text());assert all(outside(p)==expected[p] for p in pages),'Other report content changed; preserve it and require manual publication review.'
    total=0
    for m in ['image','point_ae']:
        run=O/f'{m}_seed42';t=json.loads((run/'complete.json').read_text());assert t['epoch']==99 and t['config']['epochs']==100 and t['config']['seed']==42
        for s,n in [('valid',20),('random_saved',50)]:
            e=json.loads((run/s/'results.json').read_text());assert e['complete'] and e['completed_episodes']==n and e['batch1_upstream_sampler_parity']
            assert e['checkpoint_sha256']==t['final_checkpoint_sha256'] and e['checkpoint_epoch']==99
            assert sorted(x['episode'] for x in e['episodes'])==list(range(n))
            assert all(x['delta_controller_verified'] and x['initial_state_max_abs_error']<=1e-10 and x['simulator_runtime']['robosuite']=='1.2.0' and x['simulator_runtime']['python']==sys.executable for x in e['episodes'])
            total+=n
    artifact=R/'artifacts/final_validation.json';artifact.write_text(json.dumps({'passed':True,'episodes':total,'final_epoch':99,'seed':42,'both_modalities_complete':True},indent=2)+'\n')
    paths=pages+[str(R.relative_to(L))]
    subprocess.run(['git','-C',str(L),'add','--',*paths],check=True)
    staged=subprocess.check_output(['git','-C',str(L),'diff','--cached','--name-only'],text=True).splitlines()
    assert all(p in pages or p.startswith(str(R.relative_to(L))+'/') for p in staged),'Unrelated staged changes; publication stopped.'
    subprocess.run(['git','-C',str(L),'diff','--cached','--check'],check=True)
    if staged:subprocess.run(['git','-C',str(L),'commit','-q','-m','Report100-epoch image versus frozen PointAE DP results'],check=True)
    env={**os.environ,'GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=10'}
    subprocess.run(['git','-C',str(L),'push','origin','main'],env=env,check=True)
    (O/'published_commit.txt').write_text(subprocess.check_output(['git','-C',str(L),'rev-parse','HEAD'],text=True))
if __name__=='__main__':main()
