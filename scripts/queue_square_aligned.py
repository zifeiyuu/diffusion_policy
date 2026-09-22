"""Single-GPU durable six-run training/evaluation queue."""
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import json

SCRIPTS=Path(__file__).resolve().parent
OUT=Path('/home/zxiao93/Documents/LaDiWM/results/dp_square_3050_20260920')


def status(**kw):
    tmp=OUT/'queue_status.tmp.json';tmp.write_text(json.dumps(dict(updated=time.strftime('%Y-%m-%d %H:%M:%S'),pid=os.getpid(),**kw),indent=2)+'\n');tmp.replace(OUT/'queue_status.json')


def report():
    with (OUT/'report.log').open('a') as f:
        subprocess.run([sys.executable,str(SCRIPTS/'report_square_aligned.py')],stdout=f,stderr=subprocess.STDOUT,check=True)


def command(job,args):
    status(state='running',job=job);report()
    print(time.strftime('%F %T'),job,flush=True)
    started=time.perf_counter()
    with (OUT/f'{job}.log').open('a') as f:
        p=subprocess.Popen([sys.executable,'-u',str(SCRIPTS/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT)
        status(state='running',job=job,child_pid=p.pid)
        while True:
            try:code=p.wait(timeout=60);break
            except subprocess.TimeoutExpired:report()
    duration=time.perf_counter()-started
    with (OUT/'process_wall.jsonl').open('a') as f:f.write(json.dumps(dict(job=job,wall_seconds=duration,exit_code=code))+'\n')
    if code:raise RuntimeError(f'{job} failed with exit code {code}; see {OUT/job}.log')
    report()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Queue already running')
    (OUT/'queue.pid').write_text(str(os.getpid())+'\n')
    try:
        for seed in [42,43,44]:
            for modality in ['image','point_flow']:
                name=f'{modality}_seed{seed}';run=OUT/name
                command(name+'_train',['train_square_aligned.py','--modality',modality,'--seed',str(seed)])
                for split in ['valid','random_saved']:
                    command(name+'_'+split,['eval_square_aligned.py','--run',str(run),'--split',split])
                # Reclaim only this queue's redundant optimizer snapshot after verified final evaluations.
                for split in ['valid','random_saved']:
                    assert json.loads((run/split/'results.json').read_text())['complete']
                (run/'latest.pt').unlink(missing_ok=True)
        status(state='complete',job='all six training runs and twelve evaluations');report()
    except BaseException as exc:
        status(state='failed',job=str(exc));report();raise


if __name__=='__main__':main()
