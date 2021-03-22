import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch.nn as nn
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as scipy_rotate
import pandas as pd
import torch.nn.functional as F
from string import digits
import sys
import os
import random
import torchvision.transforms.functional as TF
###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            CONSTANTS AND TYPES
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|
"""Pascal VOC Dataset Segmentation Dataloader"""
VOC_CLASSES = ('background',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

LANDCOVER_CLASS = ('building','woodlands','water')

NUM_CLASSES = len(VOC_CLASSES) + 1
def get_voc_cst():
    
    return VOC_CLASSES,NUM_CLASSES

def get_criterion(key:str,reduction='batchmean') -> dict:
    d = {'CE':nn.CrossEntropyLoss(ignore_index=21),'KL':nn.KLDivLoss(reduction = reduction, log_target = False),\
        'L1':nn.L1Loss(reduction='mean'),'MSE':nn.MSELoss()}
    return d[key]

### TYPE 
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            SAVE AND DATASETS UTILS
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

    
class Split_Dataset(Dataset):
    """
        Split a torch dataset with the same transform.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def split_dataset(dataset,percent:float) -> torch.utils.data.Dataset:
    """ 
        dataset : the dataset to split
        percent : float between 0 and 1. 
        Function use for split a dataset and use only a certain part for the supervise training.
    """
    torch.manual_seed(0)
    split = int(len(dataset)*percent)
    lengths = [split,len(dataset)-split]
    labeled, _ = random_split(dataset, lengths)
    train_full_supervised = Split_Dataset(
        labeled)
    torch.manual_seed(torch.initial_seed())
    return train_full_supervised


def my_collate(batch):
    print(len(batch))
    print(batch[0][0].size(),batch[0][1].size())
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return (data, target)

###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            METRICS FUNCTIONS
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

SMOOTH = 1e-6
def iou(outputs: torch.Tensor, labels: torch.Tensor):

    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    #iou_metric = ((iou-0.5)*2*10).floor()/10
    #iou_metric[iou_metric<0] = 0

    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch

def inter_over_union(pred, mask, num_class=21):
    """
        https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
        Inter over Union functions using numpy fast histogram.
        IoU  computed on one image or batch
        
    """
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter
    # return (area_inter/area_union) # original return of the pytorch function 
    return np.nanmean(area_inter/area_union)
def inter_over_union_all(pred, mask, num_class=21):
    """
        https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
        Inter over Union functions using numpy fast histogram.
        return iou on all class.
        
    """
    pred = np.asarray(pred, dtype=np.uint8).copy()
    mask = np.asarray(mask, dtype=np.uint8).copy()

    # 255 -> 0
    pred += 1
    mask += 1
    pred = pred * (mask > 0)

    inter = pred * (pred == mask)
    (area_inter, _) = np.histogram(inter, bins=num_class, range=(1, num_class))
    (area_pred, _) = np.histogram(pred, bins=num_class, range=(1, num_class))
    (area_mask, _) = np.histogram(mask, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_mask - area_inter
    return (area_inter/area_union) # original return of the pytorch function 

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class=21):
    label_trues = label_trues.cpu().numpy()
    label_preds = label_preds.cpu().numpy()
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def evaluate_model(model,val_loader,criterion=torch.nn.CrossEntropyLoss(ignore_index=21),nclass=21,device='cpu',plot=True):
  loss_test = []
  iou_test = []
  pixel_accuracy = []
  all_iou = []
  model.eval()
  with torch.no_grad():
    for _,(x,mask) in enumerate(val_loader):
          x = x.to(device)
          mask = mask.to(device)
          pred = model(x)
          try:
                pred = pred["out"]
          except:
                print('')
            
          loss = criterion(pred,mask)
          loss_test.append(loss.item())
          s = scores(pred.max(dim=1)[1],mask)
          IoU = inter_over_union(pred.argmax(dim=1).detach().cpu(),mask.detach().cpu())
          IoU_all = inter_over_union_all(pred.argmax(dim=1).detach().cpu(),mask.detach().cpu())
          """
            return {
              "Pixel Accuracy": acc,
              "Mean Accuracy": acc_cls,
              "Frequency Weighted IoU": fwavacc,
              "Mean IoU": mean_iu,
              "Class IoU": cls_iu,
          }
          """
          pixel_accuracy.append(s["Pixel Accuracy"])
          iou_test.append(IoU)
          all_iou.append(IoU_all)
          if plot:
              plot_pred_mask(pred.argmax(dim=1).detach().cpu()[0],mask.detach().cpu()[0],cmap=None,iou=True)
           

    

    print("Mean IOU :",np.array(iou_test).mean(),"Pixel Accuracy :",np.array(pixel_accuracy).mean(),"Loss Validation :",np.array(loss_test).mean())
    return all_iou

###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            EQUIVARIANCE UTILS FUNCTION
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

# rotate images
def rotate_image(image,angle,reshape=False):
    """
        Rotate a tensor with a certain angle.
        If true, expands the output image to make it large enough to hold the entire rotated image.
        Else it keeps the same size
        Depreciated...
    """
    #image = image.squeeze()
    if len(image.size())==3: # Case of a single image.
        axes = ((1,2))
    elif len(image.size())==4: # Case of a batch of images
        axes = ((2,3))
    else:
        print("Dimension of images must be 4 or 5.")
        return 
    im = scipy_rotate(image.numpy(),angle=angle,reshape=reshape,axes=axes)
    im_t = torch.FloatTensor(im)
    return (im_t,360-angle)

def rotate_pt(img,angle,reshape=False):
    """
        Rotate a tensor with a certain angle.
        If true, expands the output image to make it large enough to hold the entire rotated image.
        Else it keeps the same size
    """
    img = TF.rotate(img,angle=30,expand=reshape)

    return (img,360-angle)


def rotate_mask(mask,angle,reshape=False):
    """
        This function take a prediction from the model [batch_size,21,513,513] 
        and rotate, by an angle add as a parameters, the prediction.
        To make sure there is no error it is preferable to use new_angle returned by the function 'rotate_image'.
    """
    with torch.no_grad():
        if len(mask.size())==3: # Case of a single mask.
            axes = ((1,2))
        elif len(mask.size())==4: # Case of a batch of masks
            axes = ((2,3))
        else:
            print("Size must be 4 or 5.")
            return 
        m = scipy_rotate(mask.numpy(),angle=angle,reshape=reshape,axes=axes,mode='nearest')
        mask_t = torch.FloatTensor(m)
        return mask_t
    
def compute_transformations_batch(x,model,angle,reshape=False,\
                                  criterion=nn.KLDivLoss(reduction='batchmean'),Loss=None,device='cpu',plot=False):
    """
       This function compute the equivariance loss with the rotation transformation for a batch of images. 
       It also give the accuracy between the output produce by the original image and the outpute produce by the 
       transforme image.
       criterion : KL divergence / L1 Loss / MSE Loss
       Loss : 'str' ; 'KL' or 'CE' or None
       plot = True for debug
       reshape = True to allow to grow the images during the rotation to not loose the border
    """
    x = x.to(device)
    #rot_x,_= rotate_image(x.detach().cpu(),angle=angle,reshape=reshape) #Depreciated
    rot_x,_ = rotate_pt(x,angle=angle,reshape=reshape)
    logsoftmax = nn.LogSoftmax(dim=1) #LogSoftmax using instead of softmax then log.
    softmax = nn.Softmax(dim=1)
    try:
        pred_x = model(x.to(device))['out'] # a prediction of the original images.
        pred_rot = model(rot_x.to(device))['out'] # a prediction of the rotated images.
    except:
        pred_x = model(x.to(device))
        pred_rot = model(rot_x.to(device))    
    #pred_rot_x = rotate_mask(pred_x.detach().cpu(),angle,reshape=reshape) # Depreciated
    pred_rot_x,_ = rotate_pt(pred_x,angle,reshape=reshape) # Apply the rotation on the mask with the original input
    if Loss=='KL':
        loss = criterion(logsoftmax(pred_rot_x.cpu()),softmax(pred_rot.cpu())) #KL divergence between the two predictions
        loss = loss/ (pred_x.size()[2]*pred_x.size()[3]) # Divide by the number of pixel in the image. Essential for batchmean mode in KLDiv
    elif Loss == 'CE':
        loss = criterion(pred_rot.cpu(),pred_rot_x.argmax(dim=1).detach().cpu()) # Use the prediction on the original image as GTruth.  
    else:
        loss = criterion(pred_rot_x.cpu(),pred_rot.cpu()) # For loss L1, MSE…    
    acc = float(torch.sum(pred_rot_x.argmax(dim=1)==pred_rot.argmax(dim=1))/(pred_rot.size()[0]*pred_rot.size()[1]))
    # compare the pred on the original images and the pred on the rotated images put back in place
    if plot:
        class_pred = plot_equiv_mask(pred_rot.argmax(dim=1).detach().cpu()[0],pred_rot_x.argmax(dim=1).detach().cpu()[0])
        return loss,acc,class_pred
        
        
    return loss,acc  

def compute_scale_equiv_batch(x,model,size=(224,224), mode = 'nearest',\
                                  criterion=nn.KLDivLoss(reduction='batchmean'),Loss=None,device='cpu',plot=False):
    """
       This function compute the equivariance loss with the scale transformation for a batch of images. 
       It also give the accuracy between the output produce by the original image and the outpute produce by the 
       scaled image.

       size : (int,int) Size of the resized image
       mode (str) – algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. Default: 'nearest'
       criterion : KL divergence / L1 Loss / MSE Loss
       Loss : 'str' ; 'KL' or 'CE' or None
       plot = True for debug
    """
    x = x.to(device)
    original_size = x.size()[2],x.size()[3] # Save the original size of the input data 
    resized_x = F.interpolate(x,size,mode=mode) # Resize the image
    logsoftmax = nn.LogSoftmax(dim=1) #LogSoftmax using instead of softmax then log.
    softmax = nn.Softmax(dim=1)

    try:
        pred_x = model(x.to(device))['out'] # a prediction of the original images.
        pred_resized_x = model(resized_x.to(device))['out'] # a prediction of the rotated images.
    except:
        pred_x = model(x.to(device))
        pred_resized_x = model(resized_x.to(device))    
    pred_resized_x = F.interpolate(pred_resized_x,original_size,mode=mode)  # Resize the transformed input to the original size
    if Loss=='KL':
        loss = criterion(logsoftmax(pred_resized_x.cpu()),softmax(pred_x.cpu())) #KL divergence between the two predictions
        loss = loss/ (size[0]*size[0]) # Divide by the number of pixel in the image. Essential for batchmean mode in KLDiv
                                        # MAY BE WRONG !!!
    elif Loss == 'CE':
        loss = criterion(pred_resized_x.cpu(),pred_x.argmax(dim=1).detach().cpu()) # Use the prediction on the original image as GTruth.  
    else:
        loss = criterion(pred_resized_x.cpu(),pred_x.cpu()) # For loss L1, MSE…    
    acc = scores(pred_resized_x.argmax(dim=1).detach().cpu(),pred_x.argmax(dim=1).detach().cpu())["Pixel Accuracy"]
    # compare the pred on the original images and the pred on the rotated images put back in place
    if plot:
        class_pred = plot_equiv_mask(pred_resized_x.argmax(dim=1).detach().cpu()[0],pred_x.argmax(dim=1).detach().cpu()[0])
        return loss,acc,class_pred
        
        
    return loss,acc  

def eval_accuracy_equiv(model,val_loader,criterion=nn.KLDivLoss(reduction='batchmean'),\
                        nclass=21,device='cpu',Loss='KL',plot=True,angle_max=30,random_angle=False):
    """
        Function to compute the accuracy between the mask where the input had a geometric transformation 
        and the mask geometric transformed with the original input.
        random_angle -> boolean : If true a Random angle between 0 and angle_max is used for the evaluation.
        angle_max -> float : The max angle for rotate the input. 
        plot -> boolean : True plot the two masks side by side.
        Loss -> type of loss used : 'KL', 'CE' or None. 
        
    """    
    loss_test = []
    pixel_accuracy = []
    model.eval()
    with torch.no_grad():
        for i,(x,mask) in enumerate(val_loader):
            if random_angle:
                angle = np.random.randint(0,angle_max)
            else:
                angle = angle_max

            loss_equiv,acc = compute_transformations_batch(x,model,angle,reshape=False,\
                                                     criterion=criterion,Loss = Loss,\
                                                       device=device)
            loss_test.append(loss_equiv)
            pixel_accuracy.append(acc)

    m_pix_acc = np.array(pixel_accuracy).mean()
    m_loss_equiv = np.array(loss_test).mean()
    print("Mean Pixel Accuracy between masks :",m_pix_acc,"Loss Validation :",m_loss_equiv)
    return m_pix_acc, m_loss_equiv

###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            MODEL UTILS FUNCTION
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|
       
def deactivate_batchnorm(m):
    """
        Desactive batchnorm when using a torchvision model: model.apply(deactivate_batchnorm)
    """
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()
###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                           PLOT UTILS FUNCTION
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

def get_cmap() -> colors.ListedColormap:
    """
        return a cmap for pascal voc_dataset 
    """
    cmap_test = colors.ListedColormap(['black','green','blue','yellow','pink','orange','maroon','darkorange'\
                                 ,'skyblue','chocolate','azure','hotpink','tan','gold','silver','navy','white'\
                                ,'olive','beige','brown','royalblue','violet'])
    return cmap_test

def get_cmap_landcover() -> colors.ListedColormap:
    """
        return a cmap for landcover dataset
    """
    cmap_test = colors.ListedColormap(['black','blue','green'])
    return cmap_test

def plot_pred_mask(pred,mask,cmap=None,iou=True):
    """
        Function for plot the prediction, the original mask and the classes during the training.
        mask -> (size,size) device : cpu (use detach().cpu())
        pred -> (size,size) device : cpu (use detach().cpu() and argmax)
    """
    if cmap is None:
        cmap = get_cmap()
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    a.title.set_text('Ground truth')
    plt.imshow(mask,cmap=cmap,vmin=0,vmax=21) #plt.cm.get_cmap('cubehelix', 21)
    a = fig.add_subplot(1,2,2)
    plt.imshow(pred,cmap=cmap,vmin=0,vmax=21)
    a.title.set_text('Predicted mask')
    plt.show()
    class_pred = []
    class_mask = []

    for p in pred.unique():
      class_pred.append(VOC_CLASSES[int(p)])
    print("predicted classes : ",class_pred)
    for p in mask.unique():
      class_mask.append(VOC_CLASSES[int(p)])
    print("real classes : ",class_mask)
    if iou:
        IoU = inter_over_union(pred,mask)
        print('IoU on this mask :',IoU)
        
def plot_equiv_mask(rot_mask,mask,cmap=None):
    """
        Function for plot the prediction, the original mask and the classes during the training.
        rot_mask -> (size,size) device : cpu (use detach().cpu()) The mask where the input was geometric transformed
        mask -> (size,size) device : cpu (use detach().cpu() and argmax) The mask with the original input
    """
    if cmap is None:
        cmap = get_cmap()
    fig = plt.figure()
    a = fig.add_subplot(1,2,1)
    a.title.set_text('Mask with the original input')
    plt.imshow(mask,cmap=cmap,vmin=0,vmax=NUM_CLASSES) #plt.cm.get_cmap('cubehelix', 21)
    a = fig.add_subplot(1,2,2)
    a.title.set_text('Mask with the rotated input')
    plt.imshow(rot_mask,cmap=cmap,vmin=0,vmax=NUM_CLASSES)
    plt.show()
    class_pred = []
    class_mask = []
    ind_class = []

    for p in mask.unique():
        ind_class.append(int(p))
        class_pred.append(VOC_CLASSES[int(p)])
    print("predicted classes with the original input : ",class_pred)
    for p in rot_mask.unique():
      class_mask.append(VOC_CLASSES[int(p)])
    print("predicted classes with the rotated input  : ",class_mask)
    return ind_class
        
        
###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                            SAVING FUNCTION
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

def create_save_directory(path_save):
    """
        This function returned the path to a unique folder for each model running. 
    """
    files = os.listdir(path_save)
    n = [int(i) for i in files if i[0] in digits]
    if len(n)==0:
        d=0
    else:
        d = max(n)+1
    new_path = os.path.join(path_save,str(d)) #Return new path to save during training
    os.mkdir(new_path)
    return new_path

def save_curves(path,**kwargs):
    """
        path : path to save all the curves
        **kwargs : must be name_of_the_list = list
    """
    
    for name,l in kwargs.items():
        curve_name = os.path.join(path,str(name)+'.npy')
        if os.path.exists(curve_name): # If file exist save with an other name
            np.save(curve_name+str(random.randint(0,100)),np.array(l))
        else:
            np.save(curve_name,np.array(l))

def save_hparams(args,path):
    """
        Save hyperparameters of a run in a hparam.txt file
    """
    hparam = 'hparam.txt'
    fi = os.path.join(path,hparam)
    with open(fi,'w') as f:
        print(args,file=f)
    print('Hyper parameters succesfully saved in',fi)

def save_eval_angle(d_iou,save_dir):
    """
        Save the evaluation of IoU with different input angle rotation image in a file
    """
    angle = 'eval_all_angle.txt'
    fi = os.path.join(save_dir,angle)
    with open(fi,'a') as f:
        for k in d_iou.keys():
            print('Scores for datasets rotate by',k,'degrees:',file=f)
            print('   mIoU',d_iou[k]['mIoU'],'Accuracy',d_iou[k]['Accuracy'],'CE Loss',d_iou[k]['CE Loss'],file=f)
    print('Evaluation with different input rotation angle succesfully saved in',fi)

def save_model(model,save_all_ep,save_best,save_folder,model_name,ep=None,iou=None,iou_test=None):
    if save_all_ep:
        if ep is None:
            raise Exception('Saving all epochs required to have the epoch iteration.')
        save_model = model_name+'_ep'+str(ep)+'.pt'
        save = os.path.join(save_folder,save_model)
        torch.save(model,save)
    if save_best:
        if iou is None or iou_test is None:
            raise Exception('Saving best model required to pass the current IoU and the list of IoU in argument.')
        if len(iou_test)<=1:
            save_model = model_name+'.pt'
            save = os.path.join(save_folder,save_model)
            torch.save(model,save)
        else:
            if iou > max(iou_test[:len(iou_test)-1]):
                print('New saving, better IoU found')
                save_model = model_name+'.pt'
                save = os.path.join(save_folder,save_model)
                torch.save(model,save)
    else:
        save_model = model_name+'.pt'
        save = os.path.join(save_folder,save_model)
        torch.save(model,save)