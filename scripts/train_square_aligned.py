"""Train upstream DP with a fixed LaDiWM split and resumable local artifacts."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time

from square_dp_data import *
from torch.utils.data import DataLoader, RandomSampler
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel


def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),python=random.getstate())


def restore_rng(r):
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
    np.random.set_state(r['numpy']);random.setstate(r['python'])


def save_torch(path,obj):
    tmp=path.with_suffix('.tmp.pt');torch.save(obj,tmp);os.replace(tmp,path)


def train(a):
    torch.set_num_threads(4)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    output=Path(a.output) if a.output else OUT/f'{a.modality}_seed{a.seed}'
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():
        completed=json.loads((output/'complete.json').read_text())
        assert completed['config']['epochs']==a.epochs and completed['config']['modality']==a.modality
        assert completed['config']['seed']==a.seed and completed['config']['smoke']==(a.max_batches is not None)
        print('Already complete:',output,flush=True);return
    if a.deadline is not None and time.time() >= a.deadline:
        raise SystemExit(75)
    started=time.perf_counter()
    info=prepare()
    train_set=SquareDataset(a.modality,'train',info);valid_set=SquareDataset(a.modality,'valid',info)
    sample_rng=torch.Generator()
    loader=DataLoader(train_set,batch_size=64,sampler=RandomSampler(train_set,generator=sample_rng),
        generator=torch.Generator().manual_seed(a.seed),num_workers=4,pin_memory=True,persistent_workers=True)
    val_loader=DataLoader(valid_set,batch_size=64,shuffle=False,generator=torch.Generator().manual_seed(123),
        num_workers=2,pin_memory=True,persistent_workers=True)
    model=make_policy(a.modality);model.set_normalizer(train_set.normalizer());model.cuda()
    ema_model=copy.deepcopy(model);ema=EMAModel(ema_model,power=.75)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    schedule=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=len(loader)*a.epochs)
    config=dict(modality=a.modality,seed=a.seed,epochs=a.epochs,batch_size=64,train_samples=len(train_set),
        valid_samples=len(valid_set),train_frames=len(train_set.rows),valid_frames=len(valid_set.rows),
        updates_per_epoch=len(loader),horizon=16,observed_frames=2,executed_actions=8,action='7D delta OSC',
        down_dims=[512,1024,2048],diffusion_step_embed_dim=128,optimizer='AdamW',lr=1e-4,
        betas=[.95,.999],weight_decay=1e-6,lr_warmup_steps=500,lr_scheduler='cosine',
        ema_power=.75,training_noise='DDPM100 cosine epsilon',eval_sampler='DDPM100',
        checkpoint_selection='final epoch only; no rollout-based selection',
        manifest_sha256=digest(OUT/'cache/manifest.json'),parameter_count=sum(p.numel() for p in model.parameters()),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),smoke=a.max_batches is not None,
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*square*.py')})
    first_epoch=0;prior_wall=0.;epoch_sum=0.;global_step=0
    latest=output/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu')
        for k in ['modality','seed','epochs','manifest_sha256','smoke']:
            assert saved['config'][k]==config[k],k
        model.load_state_dict(saved['model']);ema_model.load_state_dict(saved['ema'])
        # Normalizer's custom loader rebuilds parameters on the checkpoint device.
        model.cuda();ema_model.cuda()
        optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['scheduler'])
        ema.optimization_step=saved['ema_step'];first_epoch=saved['epoch']+1
        prior_wall=saved['train_wall_seconds'];epoch_sum=saved['epoch_sum_seconds'];global_step=saved['global_step']
        restore_rng(saved['rng']);del saved
        record=json.loads((output/'progress.json').read_text())
        if (output/'epochs.jsonl').exists():
            records=[json.loads(x) for x in (output/'epochs.jsonl').read_text().splitlines()]
            (output/'epochs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if r['epoch']<first_epoch))
    write_json(output/'config.json',config)
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first_epoch,a.epochs):
        sample_rng.manual_seed(a.seed*100000+epoch)
        tick=time.perf_counter();model.train();losses=[]
        for bi,batch in enumerate(loader):
            batch=dict_apply(batch,lambda x:x.cuda(non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss=model.compute_loss(batch)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward()
            if bi==0 and epoch==first_epoch:
                encoder_grad=sum(float(p.grad.detach().square().sum()) for p in model.obs_encoder.parameters() if p.grad is not None)**.5
                assert np.isfinite(encoder_grad) and encoder_grad>0
                print('Trainable encoder gradient L2:',encoder_grad,flush=True)
            optimizer.step();schedule.step();ema.step(model)
            losses.append(loss.item());global_step+=1
            if a.max_batches and bi+1>=a.max_batches:break
        torch.cuda.synchronize();epoch_seconds=time.perf_counter()-tick;epoch_sum+=epoch_seconds
        model.eval();vals=[]
        # Separate validation RNG so validation cannot alter the next training epoch.
        state=rng_state();torch.manual_seed(100000+epoch)
        with torch.no_grad():
            for bi,batch in enumerate(val_loader):
                batch=dict_apply(batch,lambda x:x.cuda(non_blocking=True))
                vals.append(model.compute_loss(batch).item())
                if a.max_batches and bi+1>=a.max_batches:break
        restore_rng(state)
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=float(np.mean(vals)),
            epoch_seconds=epoch_seconds,epoch_sum_seconds=epoch_sum,global_step=global_step,
            train_wall_seconds=prior_wall+time.perf_counter()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),lr=schedule.get_last_lr()[0])
        with (output/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write_json(output/'progress.json',record)
        print(json.dumps(record),flush=True)
        deadline_reached=a.deadline is not None and time.time() >= a.deadline
        if epoch%a.save_every==0 or epoch==a.epochs-1 or deadline_reached:
            save_torch(latest,dict(model=model.state_dict(),ema=ema_model.state_dict(),optimizer=optimizer.state_dict(),
                scheduler=schedule.state_dict(),ema_step=ema.optimization_step,epoch=epoch,config=config,
                rng=rng_state(),global_step=global_step,train_wall_seconds=record['train_wall_seconds'],epoch_sum_seconds=epoch_sum))
        if deadline_reached and epoch < a.epochs-1:
            print('Deadline reached; epoch checkpoint saved. Exit75 for Slurm continuation.',flush=True)
            raise SystemExit(75)
    save_torch(output/'final_ema.pt',dict(model=ema_model.state_dict(),config=config,epoch=a.epochs-1))
    write_json(output/'complete.json',dict(**record,config=config,final_checkpoint_sha256=digest(output/'final_ema.pt'),
        total_trainer_wall_seconds=prior_wall+time.perf_counter()-started))
    # Completed jobs only need final EMA weights; keep resume files until both evals finish.


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modality',choices=['image','point_flow'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--epochs',type=int,default=3050)
    p.add_argument('--output');p.add_argument('--max-batches',type=int)
    p.add_argument('--deadline',type=float,help='Unix timestamp; save at the next epoch boundary and exit75')
    p.add_argument('--save-every',type=int,default=10)
    args=p.parse_args()
    if args.epochs<1 or args.save_every<1:p.error('epochs and save-every must be positive')
    train(args)
