import torch

from seg3d.datasets.waymo_dataset import WaymoDataset
from seg3d.models import VFE
from seg3d.models.backbones import PointTransformer
from seg3d.utils.config import cfg
from seg3d.utils.data_utils import load_data_to_gpu
from seg3d.utils.logging import get_root_logger


if __name__ == '__main__':
    logger = get_root_logger(name="test_point_transformer")

    # load data
    cfg.DATASET.POINT_CLOUD_RANGE = [-72, -72, -2, 72, 72, 4.4]
    train_dataset = WaymoDataset(cfg, '/nfs/dataset-dtai-common/waymo_open_dataset_v_1_3_2', 'validation')
    logger.info('Loaded %d train samples' % len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=2,
        num_workers=4,
        collate_fn=train_dataset.collate_batch)

    dim_point = train_dataset.dim_point
    vfe = VFE(dim_point, reduce='mean')

    batching_info = {
        0: {'max_tokens': 60, 'batching_range': (0, 60)},
        1: {'max_tokens': 120, 'batching_range': (60, 120)},
        2: {'max_tokens': 180, 'batching_range': (120, 180)},
        3: {'max_tokens': 400, 'batching_range': (180, 100000)}
    }
    window_shape = (10, 10, 4)
    model = PointTransformer(dim_point, 64, train_dataset.grid_size, train_dataset.voxel_size,
                             train_dataset.point_cloud_range, batching_info, window_shape).cuda()
    model.train()
    for step, data_dict in enumerate(train_loader):
        load_data_to_gpu(data_dict)

        points = data_dict['points'][:, 1:]
        point_voxel_ids = data_dict['point_voxel_ids']
        data_dict['voxel_features'] = vfe(points, point_voxel_ids)

        result = model(data_dict)
        print(result)
