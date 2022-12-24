import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# dataset config
__C.DATASET = edict()
__C.DATASET.USE_MULTI_SWEEPS = False
__C.DATASET.MAX_NUM_SWEEPS = 5
__C.DATASET.NUM_SWEEPS = 3
__C.DATASET.USE_CYLINDER = False
__C.DATASET.POINT_CLOUD_RANGE = [-72, -72, -2, 72, 72, 4.4]
__C.DATASET.VOXEL_SIZE = [0.1, 0.1, 0.1]
__C.DATASET.DIM_POINT = 6
__C.DATASET.USE_IMAGE_FEATURE = False
__C.DATASET.DIM_IMAGE_FEATURE = 28
__C.DATASET.NUM_CLASSES = 22
__C.DATASET.CLASS_NAMES = []
__C.DATASET.CLASS_WEIGHT = []
__C.DATASET.PALETTE = []
__C.DATASET.IGNORE_INDEX = 255

__C.DATASET.AUG_DATA = True
__C.DATASET.AUG_ROT_RANGE = [-0.78539816, 0.78539816]
__C.DATASET.AUG_SCALE_RANGE = [0.95, 1.05]
__C.DATASET.AUG_TRANSLATE_STD = 0.5
__C.DATASET.AUG_SAMPLE_RATIO = 0.95
__C.DATASET.AUG_SAMPLE_RANGE = 50.0
__C.DATASET.AUG_COLOR_DROP_RATIO = 0.5

__C.DATASET.VISUALIZE = False

# model config
__C.MODEL = edict()
__C.MODEL.SEGMENTOR = 'segformer'
__C.MODEL.LOSSES = {'ohem_ce': 1.0, 'lovasz': 1.0}
__C.MODEL.OHEM_KEEP_RATIO = 0.3
__C.MODEL.AUX_LOSS_WEIGHT = 0.4
__C.MODEL.BATCHING_INFO = [
    {
        '0': {'max_tokens': 16, 'batching_range': [0, 16]},
        '1': {'max_tokens': 64, 'batching_range': [16, 64]},
        '2': {'max_tokens': 256, 'batching_range': [64, 256]},
        '3': {'max_tokens': 800, 'batching_range': [256, 100000]}
    },
    {
        '0': {'max_tokens': 32, 'batching_range': [0, 32]},
        '1': {'max_tokens': 128, 'batching_range': [32, 128]},
        '2': {'max_tokens': 512, 'batching_range': [128, 512]},
        '3': {'max_tokens': 800, 'batching_range': [512, 100000]}
    },
    {
        '0': {'max_tokens': 64, 'batching_range': [0, 64]},
        '1': {'max_tokens': 160, 'batching_range': [64, 160]},
        '2': {'max_tokens': 384, 'batching_range': [160, 384]},
        '3': {'max_tokens': 800, 'batching_range': [384, 100000]}
    },
    {
        '0': {'max_tokens': 128, 'batching_range': [0, 128]},
        '1': {'max_tokens': 256, 'batching_range': [128, 256]},
        '2': {'max_tokens': 512, 'batching_range': [256, 512]},
        '3': {'max_tokens': 800, 'batching_range': [512, 100000]}
    }
]
__C.MODEL.WINDOW_SHAPE = [10, 10, 8]
__C.MODEL.DEPTHS = [3, 4, 8, 3]
__C.MODEL.DROP_PATH_RATE = 0.3

# training config
__C.TRAIN = edict()
__C.TRAIN.OPTIMIZER = 'adamw'
__C.TRAIN.LR = 0.001
__C.TRAIN.WEIGHT_DECAY = 0.01
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.LR_SCHEDULER = 'warmup_poly_lr'


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v
