import torch
from torch.autograd import Function

from . import sampling_ext


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        sampling_ext.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx


furthestsampling = FurthestSampling.apply


class SectorizedFurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset, num_sectors, min_points=10000):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()

        # cut into batches
        last_offset = 0
        sizes = []
        new_sizes = []
        indices = []
        for i in range(offset.shape[0]):
            size = offset[i] - last_offset
            if size < min_points:
                tmp_num_sectors = 1
            else:
                tmp_num_sectors = num_sectors
            batch_xyz = xyz[last_offset:last_offset + size]
            angle = torch.atan2(batch_xyz[:, 0], batch_xyz[:, 1])  # [0, 2*pi]
            sector_range = torch.linspace(angle.min(), angle.max() + 1e-4, tmp_num_sectors + 1)
            for s in range(tmp_num_sectors):
                indices.append(
                    torch.where((angle >= sector_range[s]) & (angle < sector_range[s + 1]))[0] + last_offset
                )
                sizes.append(indices[-1].shape[0])
            if i > 0:
                new_size = (new_offset[i] - new_offset[i - 1]).item()
            else:
                new_size = new_offset[i].item()
            new_sizes_this_batch = [new_size // tmp_num_sectors for i in range(tmp_num_sectors)]
            new_sizes_this_batch[-1] += new_size % tmp_num_sectors
            new_sizes += new_sizes_this_batch
            last_offset = offset[i]

        sizes = torch.tensor(sizes, dtype=torch.long).to(offset)
        sector_offset = sizes.cumsum(dim=0)
        new_sizes = torch.tensor(new_sizes, dtype=torch.long).to(offset)
        new_sector_offset = new_sizes.cumsum(dim=0)
        indices = torch.cat(indices).long().to(offset.device)
        sector_xyz = xyz[indices].contiguous()

        # transform to sectors
        n, b, n_max = sector_xyz.shape[0], sector_offset.shape[0], sector_offset[0]
        for i in range(1, b):
            n_max = max(sector_offset[i] - sector_offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_sector_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        sampling_ext.furthestsampling_cuda(b, n_max, sector_xyz, sector_offset.int(), new_sector_offset.int(), tmp,
                                           idx)
        idx = indices[idx.long()]
        del tmp
        del sector_xyz
        return idx


sectorized_fps = SectorizedFurthestSampling.apply
