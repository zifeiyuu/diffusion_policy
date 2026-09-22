"""Validate completed best/last artifacts and publish only this study's report."""
import gc,hashlib,json,os,re,subprocess,sys
from pathlib import Path
L=Path('/home/zxiao93/Documents/LaDiWM');O=L/'results/dp_upstream_abs100_20260922';R=L/'experiment_logs/2026-09-22/dp_upstream_abs100'
PAGES=['experiment_logs/summary/README.md','experiment_logs/others/README.md']
def outside(name):
    s=(L/name).read_text()
    if '/summary/' in name:a,b='<!-- dp-upstream-abs100:start -->','<!-- dp-upstream-abs100:end -->'
    else:
        key='experiment_logs/2026-09-22/dp_upstream_abs100/README.md';a,b=f'<!-- experiment:{key}:start -->',f'<!-- experiment:{key}:end -->'
    return hashlib.sha256(re.sub(re.escape(a)+'.*?'+re.escape(b),'',s,flags=re.S).encode()).hexdigest()
def main():
    guard=O/'publish_guard.json'
    if '--init' in sys.argv:
        guard.write_text(json.dumps({p:outside(p) for p in PAGES},indent=2)+'\n');return
    assert {p:outside(p) for p in PAGES}==json.loads(guard.read_text()),'Unrelated report content changed; leaving final publication for scoped review.'
    import torch,dill
    for m in ['image','point_ae']:
        root=O/f'{m}_seed42';t=json.loads((root/'complete.json').read_text())
        assert t['epochs_completed']==100 and t['scheduler_epochs']==3050
        ck=torch.load(t['last_checkpoint'],map_location='cpu',pickle_module=dill)
        assert {'model','ema_model','optimizer'}<=set(ck['state_dicts'])
        assert dill.loads(ck['pickles']['epoch'])==100
        del ck;gc.collect()
        for folder in [root/'last',Path(t['best_rollout'])]:
            for split,n in [('valid',20),('random_saved',50)]:
                e=json.loads((folder/split/'results.json').read_text())
                assert e['complete'] and e['completed_episodes']==n and e['batch1_upstream_sampler_parity']
                assert e['checkpoint_epoch']==(99 if folder.name=='last' else t['best_epoch'])
                assert all(x['absolute_controller_verified'] and x['initial_state_max_abs_error']<=1e-10 and x['simulator_runtime']['python']==sys.executable for x in e['episodes'])
    (R/'artifacts/final_validation.json').write_text(json.dumps({'passed':True,'last_full_checkpoint_contains':['model','ema_model','optimizer'],'epochs_completed':100,'scheduler_epochs':3050,'both_best_and_last_evaluated':True},indent=2)+'\n')
    paths=PAGES+[str(R.relative_to(L))]
    staged=subprocess.check_output(['git','-C',str(L),'diff','--cached','--name-only'],text=True).splitlines()
    assert not staged,'Existing staged changes; publication stopped.'
    subprocess.run(['git','-C',str(L),'add','--',*paths],check=True)
    subprocess.run(['git','-C',str(L),'diff','--cached','--check'],check=True)
    subprocess.run(['git','-C',str(L),'commit','-m','Report upstream absolute DP at 100 epochs with best and last checkpoints'],check=True)
    subprocess.run(['git','-C',str(L),'push','origin','main'],env={**os.environ,'GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=10'},check=True)
    (O/'published_commit.txt').write_text(subprocess.check_output(['git','-C',str(L),'rev-parse','HEAD'],text=True))
if __name__=='__main__':main()
