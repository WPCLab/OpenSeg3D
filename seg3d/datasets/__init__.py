import random
from functools import partial

import numpy as np

import torch
from torch.utils.data import DataLoader

from seg3d.utils.distributed import get_dist_info
from seg3d.datasets.samplers.distributed_sampler import DistributedSampler


def build_dataloader(dataset, batch_size, dist=False, num_workers=4, collate_fn=None, seed=None, training=True):
    if dist:
        rank, world_size = get_dist_info()

        if training:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=True, seed=seed)
        else:
            sampler = DistributedSampler(
                dataset, world_size, rank, shuffle=False, seed=seed)
    else:
        sampler = None

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        shuffle=(sampler is None) and training, collate_fn=collate_fn,
        worker_init_fn=init_fn, drop_last=False, sampler=sampler, timeout=0
    )

    return dataset, dataloader, sampler


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.
    The seed of each worker equals to num_worker * rank + worker_id + user_seed
    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
