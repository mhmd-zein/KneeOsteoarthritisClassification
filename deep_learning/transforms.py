from configs import config
import os 
from conventional.transforms import EnsureNotInvertedd
from monai.transforms import (
    Compose,
    LoadImaged,
    Rotate90d,
    ToNumpyd,
    ToTensord,
    RepeatChanneld,
    NormalizeIntensityd,
    RandFlipd,
    RandRotated,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    Flipd
    )

train_transforms = Compose([ # used for training set only, includes data augmentation transforms
    LoadImaged(keys='image', ensure_channel_first=True),
    Rotate90d(keys='image', k=-1),
    Flipd('image', spatial_axis=1),
    ToNumpyd('image'),
    EnsureNotInvertedd(keys='image', output_key='image', pos_refs=[os.path.join(config.data['path'],pos_ref) for pos_ref in config.transforms['pos_refs']], neg_refs=[os.path.join(config.data['path'],neg_ref) for neg_ref in config.transforms['neg_refs']]),
    ToTensord('image', device='cuda'),
    RepeatChanneld('image', 3),
    RandFlipd('image',prob=0.5,spatial_axis=1), # horizontal
    RandRotated('image', range_x=0.2, prob=0.5),
    NormalizeIntensityd('image'),
    RandAdjustContrastd('image', gamma=(0.25,2.25), prob=0.5),
    RandGaussianSmoothd('image', prob=0.5, sigma_x=(0.5,2), sigma_y=(0.5,2), sigma_z=(0.5,2)),
    RandGaussianNoised('image', prob=0.5, std=0.3),
    NormalizeIntensityd('image'),
])

test_transforms = Compose([ # used for validation and testing sets
    LoadImaged(keys='image', ensure_channel_first=True),
    Rotate90d(keys='image', k=-1),
    Flipd('image', spatial_axis=1),
    ToNumpyd('image'),
    EnsureNotInvertedd(keys='image', output_key='image', pos_refs=[os.path.join(config.data['path'],pos_ref) for pos_ref in config.transforms['pos_refs']], neg_refs=[os.path.join(config.data['path'],neg_ref) for neg_ref in config.transforms['neg_refs']]),
    ToTensord('image', device='cuda'),
    RepeatChanneld('image', 3),
    NormalizeIntensityd('image'),
])