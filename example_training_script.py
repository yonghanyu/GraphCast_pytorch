import torch
import os
import sys
import einops
import datetime

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


sys.path.append('project_root/src')
from graphcast import GraphCast
from era_loader import ERA5Dataset, ERA5DataConfig
from scheduler import cosine_lr, linear_lr_decay

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




def main():
    pressure_data_dir = "path/to/pressure/data"
    surface_data_dir = "path/to/surface/data"
    data_mean = 'path/to/mean'
    data_std = 'path/to/std'
    data_diff = 'path/to/diff_std'
    cfg = ERA5DataConfig("config/data_config.yaml")

    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(minutes=30))
    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = torch.device(f"cuda:{local_rank}")
    batch_size = 1
    lr = 3e-4
    warmup_step = 1000
    training_horizon = 3
    n_epochs = 100


    dataset = ERA5Dataset(cfg,
            pressure_data_dir,
            surface_data_dir,
            data_mean_path=data_mean,
            data_std_path=data_std,
            data_diff_std_path=data_diff,
            epoch_len=training_horizon,
            add_time_feature=True)


    latitudes = dataset.latitude
    longitudes = dataset.longitude

    in_mean = torch.from_numpy(dataset.mean)
    in_std = torch.from_numpy(dataset.std)
    forcing_mean = torch.from_numpy(dataset.forcing_mean)
    forcing_std = torch.from_numpy(dataset.forcing_std)
    out_std = torch.from_numpy(dataset.diff_std).to(device)
    loss_weight = dataset.loss_weight.to(device)
    area_weight = dataset.area_weight.to(device)

    in_dim = (in_mean.shape[1] + forcing_mean.shape[1]) * 2
    out_dim = in_mean.shape[1]

    model = GraphCast(latitudes, longitudes,
                       in_dim=in_dim,
                       out_dim=out_dim,
                       n_blocks=11,
                       n_mp_per_blocks=1,
                       nround_mp=14,
                       mesh_size=6,
                       )
    model.to(device)
    model = DDP(model, device_ids=None, output_device=None)

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, num_replicas=world_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=12, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))


    accum_loss = 0
    accum_step = 0

    scaler = torch.amp.grad_scaler.GradScaler()
    total_steps = len(train_loader) * n_epochs
    schedular = cosine_lr(optimizer, lr, warmup_step, total_steps)

    for epoch in range(n_epochs):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        for i, batch_data in enumerate(train_loader):
            step = epoch * len(train_loader) + i
            # [b 2 c lat lon]
            data = batch_data['data']
            forcing = batch_data['forcing']
            x_out = batch_data['target'][:, 1] # [b c lat lon] x_t+1

            with torch.no_grad():
                data_n = (data - in_mean) / in_std
                forcing = (forcing - forcing_mean) / forcing_std

            x_in = torch.cat([data_n, forcing], dim=2)

            x_in = x_in.to(device)
            x_in = einops.rearrange(x_in, 'b t c n m -> b (t c) n m') # concat [x_t-1, x_t]
            data = data[:, 1] # [b c lat lon] x_t
            data = data.to(device)
            x_out = x_out.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                optimizer.zero_grad()
                x_pred = model(x_in) # [b c lat lon] x_t+1
                x_pred = x_pred * out_std + data
                loss = ((x_pred - x_out) / out_std) ** 2
                loss = loss * loss_weight * area_weight
                loss = torch.nansum(loss, dim=1).mean()


            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            new_lr = schedular(step)

            accum_loss += loss.item()

            if i > 0 and i % 5 == 0 and global_rank == 0:
                loss_p = accum_loss / (i - accum_step)
                accum_step = i
                accum_loss = 0
                print(f'Epoch: {epoch}, batch: {i}, Loss: {loss_p}, LR: {new_lr}')


        if (global_rank == 0 and epoch % 1 == 0) or epoch == n_epochs - 1:
            out_name = 'model' + str(epoch) + '.pth'
            checkpoint = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, out_name)


if __name__ == '__main__':
    main()
