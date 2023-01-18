import os
import time
import argparse

import torch
import torch.optim

from seg3d.datasets.waymo_dataset import WaymoDataset
from seg3d.datasets import build_dataloader
from seg3d.models.builder import build_segmentor, build_criterion, build_optimizer, build_scheduler
from seg3d.core.evaluation import IOUMetric
from seg3d.ops import knn_query
from seg3d.utils.logging import get_root_logger
from seg3d.utils.distributed import init_dist, get_dist_info
from seg3d.utils.config import cfg, cfg_from_file
from seg3d.utils.data_utils import load_data_to_gpu
from seg3d.utils.pointops_utils import get_voxel_centers
from seg3d.utils.random import set_random_seed, init_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a 3d segmentor')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--data_dir', type=str, help='the data directory')
    parser.add_argument('--save_dir', type=str, help='the saved directory')
    parser.add_argument('--pretrained_path', type=str, default=None, help='pretrained_path')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--cudnn_benchmark', action='store_true', default=False, help='whether to use cudnn')
    parser.add_argument('--deterministic', action='store_true', default=False, help='whether to use deterministic')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--no_validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--eval_epoch_interval', default=2, type=int)
    parser.add_argument('--log_iter_interval', default=10, type=int)
    parser.add_argument('--auto_resume', action='store_true', help='resume from the latest checkpoint automatically')
    args = parser.parse_args()

    return args


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def save_checkpoint(model, optimizer, lr_scheduler, save_dir, epoch, logger):
    logger.info('Save checkpoint at epoch %d' % epoch)
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.state_dict()

    checkpoint = {
        "model": model_state,
        'optimizer': optimizer.state_dict(),
        "epoch": epoch,
        'lr_scheduler': lr_scheduler.state_dict()
    }

    torch.save(checkpoint, os.path.join(save_dir, 'epoch_%s.pth' % str(epoch)))
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))


def compute_loss(pred_result, data_dict, criterion):
    loss = 0

    point_gt_labels = data_dict['point_labels']
    point_pred_labels = pred_result['point_out']
    for loss_func, loss_weight in criterion:
        loss += loss_func(point_pred_labels, point_gt_labels) * loss_weight

    if 'voxel_out' in pred_result:
        voxel_gt_labels = data_dict['voxel_labels']
        voxel_pred_labels = pred_result['voxel_out']
        for loss_func, loss_weight in criterion:
            loss += loss_func(voxel_pred_labels, voxel_gt_labels) * loss_weight

    if 'aux_voxel_out' in pred_result:
        with torch.no_grad():
            voxel_coords = pred_result['voxel_coords']
            aux_voxel_coords = pred_result['aux_voxel_coords']
            voxel_centers = get_voxel_centers(voxel_coords[:, 1:], 1.0, cfg.DATASET.VOXEL_SIZE,
                                              cfg.DATASET.POINT_CLOUD_RANGE)
            aux_voxel_centers = get_voxel_centers(aux_voxel_coords[:, 1:], 8.0, cfg.DATASET.VOXEL_SIZE,
                                                  cfg.DATASET.POINT_CLOUD_RANGE)

            voxel_id_offset, count = [], 0
            aux_voxel_id_offset, aux_count = [], 0
            for i in range(data_dict['batch_size']):
                count += torch.sum(voxel_coords[:, 0] == i)
                aux_count += torch.sum(aux_voxel_coords[:, 0] == i)
                voxel_id_offset.append(count)
                aux_voxel_id_offset.append(aux_count)
            voxel_id_offset = torch.tensor(voxel_id_offset, device=voxel_centers.device).int()
            aux_voxel_id_offset = torch.tensor(aux_voxel_id_offset, device=aux_voxel_centers.device).int()
            query_idx, _ = knn_query(1, voxel_centers, aux_voxel_centers, voxel_id_offset, aux_voxel_id_offset)
            aux_voxel_gt_labels = voxel_gt_labels[query_idx.squeeze().long()]

        aux_voxel_pred_labels = pred_result['aux_voxel_out']
        for loss_func, loss_weight in criterion:
            loss += cfg.MODEL.AUX_LOSS_WEIGHT * loss_func(aux_voxel_pred_labels, aux_voxel_gt_labels) * loss_weight

    return loss


def evaluate(args, data_loader, model, criterion, class_names, epoch, logger):
    iou_metric = IOUMetric(class_names)
    model.eval()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            result = model(data_dict)

        loss = compute_loss(result, data_dict, criterion)

        if step % args.log_iter_interval == 0:
            logger.info(
                'Evaluate on epoch %d - Iter [%d/%d] loss: %f' % (epoch, step, len(data_loader), loss.cpu().item()))

        pred_labels = torch.argmax(result['point_out'], dim=1).cpu()
        gt_labels = data_dict['point_labels'].cpu()
        iou_metric.add(pred_labels, gt_labels)

    metric_result = iou_metric.get_metric()
    logger.info('Metrics on validation dataset: %s' % str(metric_result))


def train_epoch(args, data_loader, model, criterion, optimizer, lr_scheduler, epoch, logger):
    model.train()
    for step, data_dict in enumerate(data_loader, 1):
        load_data_to_gpu(data_dict)

        result = model(data_dict)

        loss = compute_loss(result, data_dict, criterion)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        if step % args.log_iter_interval == 0:
            try:
                cur_lr = float(optimizer.lr)
            except:
                cur_lr = optimizer.param_groups[0]['lr']
            logger.info(
                'Train - Epoch [%d/%d] Iter [%d/%d] lr: %f, loss: %f' % (epoch, args.epochs, step, len(data_loader),
                                                                         cur_lr, loss.cpu().item()))


def train_segmentor(args, start_epoch, data_loaders, train_sampler, class_names, model,
                    criterion, optimizer, lr_scheduler, rank, logger):
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # current epoch
        cur_epoch = epoch + 1

        # train for one epoch
        train_epoch(args, data_loaders['train'], model, criterion, optimizer, lr_scheduler, cur_epoch, logger)

        # save checkpoint
        if rank == 0 and args.auto_resume:
            save_checkpoint(model, optimizer, lr_scheduler, args.save_dir, cur_epoch, logger)

        # evaluate on validation set
        if not args.no_validate and cur_epoch % args.eval_epoch_interval == 0:
            evaluate(args, data_loaders['val'], model, criterion, class_names, cur_epoch, logger)


def main():
    # parse args
    args = parse_args()

    # whether to distributed training
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher)
        # gpu_ids is used to calculate iter when resuming checkpoint
        rank, world_size = get_dist_info()

    # set cudnn_benchmark
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # create saved directory
    os.makedirs(args.save_dir, exist_ok=True)

    # create logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.save_dir, f'{timestamp}.log')
    logger = get_root_logger(name="seg3d", log_file=log_file)

    # set random seed
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)

    # config
    cfg_from_file(args.cfg_file)
    logger.info(cfg)

    # load data
    train_dataset = WaymoDataset(cfg, os.path.join(args.data_dir, 'training'), mode='training')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        dist=distributed,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_batch,
        seed=seed,
        training=True)
    data_loaders = {'train': train_loader}
    logger.info('Loaded %d train samples' % len(train_dataset))

    if not args.no_validate:
        val_dataset = WaymoDataset(cfg, os.path.join(args.data_dir, 'validation'), mode='validation')
        val_set, val_loader, sampler = build_dataloader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            dist=distributed,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_batch,
            seed=seed,
            training=False)
        data_loaders['val'] = val_loader
        logger.info('Loaded %d validation samples' % len(val_dataset))

    # define model
    model = build_segmentor(cfg, train_dataset)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # load pretrained model
    if args.pretrained_path:
        loc_type = torch.device('cpu') if distributed else None
        pretrained = torch.load(args.pretrained_path, map_location=loc_type)
        model.load_state_dict(pretrained['model'], strict=False)
        logger.info('Loaded pretrained model from %s' % args.pretrained_path)

    # optimizer and learning rate scheduler
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer, args.epochs, len(train_loader))

    # resume from checkpoint
    start_epoch = 0
    latest_checkpoint = os.path.join(args.save_dir, 'latest.pth')
    if args.auto_resume and os.path.isfile(latest_checkpoint):
        loc_type = torch.device('cpu') if distributed else None
        checkpoint = torch.load(latest_checkpoint, map_location=loc_type)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logger.info('Resume from epoch %d' % start_epoch)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank % torch.cuda.device_count()],
                                                          find_unused_parameters=False)

    # loss function
    criterion = build_criterion(cfg, train_dataset)

    # train and evaluation
    train_segmentor(args, start_epoch, data_loaders, train_sampler, train_dataset.class_names,
                    model, criterion, optimizer, lr_scheduler, rank, logger)


if __name__ == '__main__':
    main()
