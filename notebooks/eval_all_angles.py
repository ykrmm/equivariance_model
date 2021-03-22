#%matplotlib inline
import argparse
import os
from os.path import isfile,join
import random
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import torch.utils.data as tud
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from collections import Counter
from sklearn.metrics import jaccard_score
import pickle
import sys
sys.path.insert(1, '../utils')
sys.path.insert(1, '../datasets')
sys.path.insert(1, '../utils_on_gpu')
import coco_utils as cu
import my_datasets as mdset
import eval_train as ev
import utils as U
import find_best_model as fbm   


# DATASETS
dataroot_voc = '/share/DEEPLEARNING/datasets/voc2012'
dataroot_sbd = '/share/DEEPLEARNING/datasets/sbd'
dataroot_coco = '/share/DEEPLEARNING/datasets/coco'
dataroot_landcover = '/share/DEEPLEARNING/datasets/landcover'
dataroot_coco2voc = '/users/k/karmimy/data/coco2voc'
VOC = False

nw = 4 
pm = True
#MODEL SAVE AND LOAD 
load_dir = '/share/homes/karmimy/equiv/save_model' # Saved model dir
#exp = 'rot_equiv_lc' # Exp dir
exp = 'rot_equiv_lc' 
#model_name = 'rot_equiv_lc.pt' # saved model name
model_name = 'rot_equiv_lc.pt'
folder_model = join(load_dir,exp) 
#fcn= True
#pretrained=True

# GPU 
gpu = 0
# EVAL PARAMETERS
bs = 2  

# LOSS 
criterion_supervised = nn.CrossEntropyLoss(ignore_index=21) # On ignore la classe border.
Loss = 'KL' # Loss = 'KL' or 'CE' or None for L1,MSE…
criterion_unsupervised = U.get_criterion(Loss)


# SEARCH FOR A PARTICULAR MODEL 
rotate = False # random rotation during training
scale = False
split = True # split the supervised dataset
split_ratio = 0.3
batch_size = 4
pi_rotate = False

#scale_factor = (0.2,0.8)
#size_img = (420,420) 
#size_crop = (380,380)

# DEVICE
# Decide which device we want to run on
device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
print("device :",device)

model_dir = '/share/homes/karmimy/equiv/save_model/rot_equiv_lc/11'

model = torch.load(join(model_dir,model_name),map_location=device)

if VOC :
        num_classes = 21
        train_dataset_VOC = mdset.VOCSegmentation(dataroot_voc,year='2012', image_set='train', \
                download=True,rotate=rotate,scale=scale,size_img=size_img,size_crop=size_crop)
        test_dataset = mdset.VOCSegmentation(dataroot_voc,year='2012', image_set='val', download=True)
        train_dataset_SBD = mdset.SBDataset(dataroot_sbd, image_set='train_noval',mode='segmentation',\
                rotate=rotate,scale=scale,size_img=size_img,size_crop=size_crop)
        train_dataset = tud.ConcatDataset([train_dataset_VOC,train_dataset_SBD])
        
else:
        num_classes = 4
        print('Loading Landscape Dataset')
        train_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="trainval",\
            rotate=rotate)#,size_img=size_img,size_crop=size_crop)
        test_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="test")
        test_dataset_no_norm =  mdset.LandscapeDataset(dataroot_landcover,image_set="test",normalize=False)
        print('Success load Landscape Dataset')
dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=bs,num_workers=nw,\
        pin_memory=pm,shuffle=True,drop_last=True)#,collate_fn=U.my_collate)
dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=nw,pin_memory=pm,\
        batch_size=bs)



l_angles = [320,330,340,350,0,10,20,30,40]
l_iou = []
for angle in l_angles:
    test_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="test",fixing_rotate=True,angle_fix=angle)
    dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=nw,pin_memory=pm,\
        batch_size=bs)
    state = ev.eval_model(model,dataloader_val,device=device,num_classes=4)
    iou = state.metrics['mean IoU']
    acc = state.metrics['accuracy']
    loss = state.metrics['CE Loss']
    print('EVAL FOR ANGLE',angle,': IOU',iou,', ACCURACY',acc,', LOSS',loss)
    l_iou.append(state.metrics['mean IoU'])
