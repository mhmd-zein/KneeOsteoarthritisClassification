import os 
from configs import config
from monai.data import Dataset, DataLoader
from conventional.transforms import conventional_transforms
import torch
import random
import numpy as np
import monai 

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)
    
def get_data(data_path, mode, balance=False):
    data_path = os.path.join(data_path,mode)
    dataset = []
    classes_total = [len(os.listdir(os.path.join(data_path, class_folder))) for class_folder in os.listdir(data_path)]
    min_class = min(classes_total)
    classes_counter = [0,0,0,0,0]
    for class_folder in os.listdir(data_path):
        for image_path in os.listdir(os.path.join(data_path,class_folder)):
            classes_counter[int(class_folder)]+=1
            if balance and classes_counter[int(class_folder)]>min_class:
                continue
            dataset.append({'image_path': os.path.join(data_path,class_folder,image_path), 'image': os.path.join(data_path,class_folder,image_path), 'label': int(class_folder)})
    return dataset

def get_loader(data_path, transforms = None, mode = 'train', balance=False, shuffle = False, batch_size = 1, seed=42):
    set_seed(seed)
    data = get_data(data_path, mode, balance)
    dataset = Dataset(data, transforms)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
    return dataloader