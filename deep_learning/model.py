import monai
import torch
from data.dataset import train_loader, val_loader, test_loader
from deep_learning.engine import Engine
from utils import set_seed
from torchvision.models import efficientnet_b0, efficientnet_b4, vgg16
from monai.losses import FocalLoss
from torch.nn import CrossEntropyLoss, Linear
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from data.dataset import get_loader
from configs import config
from deep_learning.transforms import train_transforms, test_transforms
set_seed(42)

dataset_path = config.data['path']
train_loader = get_loader(dataset_path, transforms=train_transforms, mode='train', balance=False, shuffle=config.data['shuffle'], batch_size=config.data['batch_size'], seed=config.seed)
val_loader = get_loader(dataset_path, transforms=test_transforms, mode='val', shuffle=False, batch_size=config.data['batch_size'])
test_loader = get_loader(dataset_path, transforms=test_transforms, mode='test', shuffle=False, batch_size=config.data['batch_size'])

network = efficientnet_b4(pretrained=True,stochastic_depth_prob=0.5,dropout=0.5)
network.classifier[-1] = Linear(network.classifier[-1].in_features, 5)
optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)
# scheduler, step = StepLR(optimizer, step_size=10, gamma=0.5), 'epoch'
scheduler, step = CosineAnnealingWarmRestarts(optimizer, T_0=5*len(train_loader)), 'batch'
loss = FocalLoss(gamma=2)
engine = Engine(network, optimizer, loss, train_loader, val_loader, scheduler, step)