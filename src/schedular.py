import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    lr = base_lr * (step + 1) / warmup_length
    return lr 



def linear_lr_decay(optimizer, base_lr, total_steps, start_decay_step, decay_end_lr=0.):
    def _lr_adjuster(step):
        if step < start_decay_step:
            lr = base_lr
        else:
            e = step - start_decay_step
            es = total_steps - start_decay_step
            # linear decay if power == 1; polynomial decay otherwise;
            decay = 1 - (e/es)
            lr = decay * (base_lr - decay_end_lr) + decay_end_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, total_steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = total_steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster