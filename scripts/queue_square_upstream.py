"""Sequential official-environment training, best/last evaluations, and reporting."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parents[1];O=Path('/home/zxiao93/Documents/LaDiWM/results/dp_upstream_abs100_20260922')
def status(state,job,pid=None):
    (O/'queue_status.json').write_text(json.dumps(dict(state=state,job=job,pid=pid,updated=time.time()),indent=2)+'\n')
def report():
    with (O/'report.log').open('a') as f:subprocess.run([sys.executable,'scripts/report_square_upstream.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
def run(name,args):
    with (O/f'{name}.log').open('a') as f:
        p=subprocess.Popen([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT);status('running',name,p.pid)
        while p.poll() is None:
            report()
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:pass
        if p.returncode:raise RuntimeError(f'{name} failed: {p.returncode}')
def main():
    os.chdir(D);O.mkdir(parents=True,exist_ok=True)
    lock=(O/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='offline')
    os.environ['LD_LIBRARY_PATH']=str(Path.home()/'.mujoco/mujoco210/bin')+':/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
    try:
        run('prepare',['scripts/prepare_square_abs100.py'])
        assert (O/'preflight.json').exists(),'Both observation adapters must pass actual loss and rollout preflight first'
        for m in ['image','point_ae']:
            run(m+'_train',['scripts/train_square_upstream.py','--modality',m])
            root=O/f'{m}_seed42'
            for candidate in sorted((root/'rollouts').glob('epoch_*')):
                for split in ['valid','random_saved']:
                    run(m+'_'+candidate.name+'_'+split,['scripts/eval_square_upstream.py','--run',str(candidate),'--split',split])
            candidates=[]
            for candidate in (root/'rollouts').glob('epoch_*'):
                e=json.loads((candidate/'random_saved/results.json').read_text())
                candidates.append((e['success_rate'],e['checkpoint_epoch'],str(candidate)))
            best=max(candidates,key=lambda x:(x[0],-x[1]))
            record=json.loads((root/'complete.json').read_text())
            record.update(best_sr=best[0],best_epoch=best[1],best_rollout=best[2],selection='Random50 SR in robosuite1.4.1; earliest epoch breaks ties')
            (root/'complete.json').write_text(json.dumps(record,indent=2)+'\n')
            for s in ['valid','random_saved']:run(m+'_last_'+s,['scripts/eval_square_upstream.py','--run',str(O/f'{m}_seed42/last'),'--split',s])
        status('complete','Both models; best and last checkpoints evaluated');report()
        with (O/'publish.log').open('a') as f:
            subprocess.run([sys.executable,'scripts/publish_square_upstream.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
    except BaseException as e:status('failed',str(e));report();raise
if __name__=='__main__':main()
