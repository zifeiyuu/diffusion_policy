"""Durable one-GPU two-model100-epoch experiment queue."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
D=Path(__file__).resolve().parents[1];L=Path('/home/zxiao93/Documents/LaDiWM');O=L/'results/dp_image_pointae_100_20260922'
os.environ.update(DP_OUTPUT_ROOT=str(O),DP_SIM_PYTHON=sys.executable,DP_SIM_WORKER='square_dp_native_worker.py',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
os.environ['LD_LIBRARY_PATH']=str(Path.home()/'.mujoco/mujoco210/bin')+':/usr/lib/x86_64-linux-gnu:'+os.environ.get('LD_LIBRARY_PATH','')
os.chdir(D)
def status(state,job,pid=None):
 p=O/'queue_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps({'state':state,'job':job,'child_pid':pid,'updated':time.time()},indent=2)+'\n');q.replace(p)
def report():
 with (O/'report.log').open('a') as f:subprocess.run([sys.executable,'scripts/report_square_100.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
def run(job,args):
 with (O/f'{job}.log').open('a') as f:
  tick=time.perf_counter();p=subprocess.Popen([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT);status('running',job,p.pid)
  while p.poll() is None:
   report()
   try:p.wait(timeout=45)
   except subprocess.TimeoutExpired:pass
  with (O/'process_wall.jsonl').open('a') as t:t.write(json.dumps({'job':job,'wall_seconds':time.perf_counter()-tick,'exit_code':p.returncode})+'\n')
  if p.returncode:raise RuntimeError(f'{job}: exit{p.returncode}')
  report()
def main():
 p=argparse.ArgumentParser();p.add_argument('--adopt-image-pid',type=int);a=p.parse_args();O.mkdir(parents=True,exist_ok=True)
 lock=(O/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  if a.adopt_image_pid:
   status('running','image_train',a.adopt_image_pid)
   while not (O/'image_seed42/complete.json').exists():
    # A dead initial trainer must not leave the queue waiting indefinitely.
    try:os.kill(a.adopt_image_pid,0)
    except ProcessLookupError:raise RuntimeError('Adopted image trainer exited before completion')
    report();time.sleep(45)
  for m in ['image','point_ae']:
   folder=O/f'{m}_seed42'
   if not (folder/'complete.json').exists():run(m+'_train',['scripts/train_square_aligned.py','--modality',m,'--seed','42','--epochs','100','--save-every','10'])
   for split in ['valid','random_saved']:run(m+'_'+split,['scripts/eval_square_aligned.py','--run',str(folder),'--split',split])
   assert all(json.loads((folder/s/'results.json').read_text())['complete'] for s in ['valid','random_saved'])
   (folder/'latest.pt').unlink(missing_ok=True)
  status('complete','two100-epoch models and four evaluations');report()
 except BaseException as e:
  status('failed',str(e));report();raise
if __name__=='__main__':main()
