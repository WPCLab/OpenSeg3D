from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    `torch.utils.data.DistributedSampler`.
    Args:
        dataset (Dataset): the dataset will be loaded.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, world_size is retrieved from the
            current distributed group.
        rank (int, optional):  Rank of the current process within num_replicas.
            By default, rank is retrieved from the current distributed group.
        shuffle (bool): If True (default), sampler will shuffle the indices.
        seed (int): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
    """

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed=0) -> None:
        super(DistributedSampler, self).__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        self.seed = seed

    def __iter__(self) -> Iterator:
        """
         Yields:
            Iterator: iterator of indices for rank.
        """
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            # When :attr:`shuffle=True`, this ensures all replicas
            # use a different random ordering for each epoch.
            # Otherwise, the next iteration of this sampler will
            # yield the same ordering.
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)