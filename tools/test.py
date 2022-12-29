import os
import time
import argparse
from tqdm import tqdm

import torch
import torch.optim
import torch.nn.functional as F

from seg3d.datasets.waymo_dataset import WaymoDataset
from seg3d.datasets import build_dataloader
from seg3d.datasets.transforms.test_time_aug import MultiScaleFlipAug
from seg3d.models.builder import build_segmentor
from seg3d.utils.config import cfg_from_file, cfg
from seg3d.utils.logging import get_logger
from seg3d.utils.submission import construct_seg_frame, write_submission_file
from seg3d.utils.data_utils import load_data_to_gpu

from waymo_open_dataset.protos import segmentation_metrics_pb2


def parse_args():
    parser = argparse.ArgumentParser(description='Test a 3d segmentor')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--tta', action='store_true', default=False, help='whether to use tta')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='whether to use cudnn')
    parser.add_argument('--log_iter_interval', default=5, type=int)
    args = parser.parse_args()

    return args

def semseg_for_one_frame(args, model, data_dict, augmentor):
    points_ri = data_dict['points_ri']
    frame_id = data_dict['filename'][0]

    if args.tta:
        point_out_list = []
        aug_data_list = augmentor(data_dict)
        for i in range(len(aug_data_list)):
            aug_data = aug_data_list[i]
            load_data_to_gpu(aug_data)
            with torch.no_grad():
                result = model(aug_data)
            point_out = F.softmax(result['point_out'], dim=1)
            point_out_list.append(point_out)
        point_outs = torch.stack(point_out_list, dim=0)
        point_out = torch.mean(point_outs, dim=0)
    else:
        load_data_to_gpu(data_dict)
        with torch.no_grad():
            result = model(data_dict)
        point_out = result['point_out']
    pred_labels = torch.argmax(point_out, dim=1).cpu()

    seg_frame = construct_seg_frame(pred_labels, points_ri, frame_id)
    return seg_frame

def inference(args, augmentor, data_loader, model, logger):
    logger.info('Inference start!')
    model.eval()
    segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()
    for step, data_dict in enumerate(tqdm(data_loader), 1):
        segmentation_frame = semseg_for_one_frame(args, model, data_dict, augmentor)
        segmentation_frame_list.frames.append(segmentation_frame)

    submission_file = os.path.join(args.save_dir, 'wod_test_set_pred_semantic_seg.bin')
    write_submission_file(segmentation_frame_list, submission_file)

    logger.info('Inference finished!')

def main():
    # parse args
    args = parse_args()

    # set cudnn_benchmark
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.save_dir, f'{timestamp}.log')
    logger = get_logger("seg3d", log_file)

    # config
    cfg_from_file(args.cfg_file)
    logger.info(cfg)

    # load data
    test_dataset = WaymoDataset(cfg, args.data_dir, 'testing', test_mode=True)
    logger.info('Loaded %d testing samples' % len(test_dataset))

    test_set, test_loader, sampler = build_dataloader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        dist=False,
        num_workers=args.num_workers,
        training=False)

    # data augmentor
    augmentor = MultiScaleFlipAug(dataset=test_dataset,
                                  scales=[0.95, 1.0, 1.05],
                                  angles=[-0.78539816, 0, 0.78539816],
                                  flip_x=True, flip_y=True)

    # define model
    model = build_segmentor(cfg, test_dataset).cuda()
    checkpoint = torch.load(os.path.join(args.save_dir, 'latest.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # inference
    inference(args, augmentor, test_loader, model, logger)


if __name__ == '__main__':
    main()
