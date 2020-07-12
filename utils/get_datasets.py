import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import numpy as np
import matplotlib.pyplot as plt
import pickle
import eval_train as ev
import PIL
from utils import * 

dataroot = '/data/voc2012'
SAVE_DIR = '/data/model'
size = (513,513)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



def get_dataset_val(batch_size,angle):
    def to_tensor_target(img):
        img = np.array(img)
        # border
        img[img==255] = 0 # border = background 
        return torch.LongTensor(img)

    def rotate_pil(img,fill=0):
        img = TF.rotate(img,angle=angle,fill=fill)
        return img

    def hflip(img):
        img = TF.hflip(img)
        return img
    transform_input = transforms.Compose([
                                    transforms.Resize(size),
                                    transforms.Lambda(rotate_pil),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=MEAN, std=STD),
                                        ])

    transform_mask = transforms.Compose([
                                    transforms.Resize(size),
                                    transforms.Lambda(rotate_pil),
                                    transforms.Lambda(to_tensor_target)
                                    ])
    val_dataset = dset.VOCSegmentation(dataroot,year='2012', image_set='val', download=True,
                                     transform= transform_input,
                                     target_transform= transform_mask)
    
    dataloader_val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return dataloader_val

