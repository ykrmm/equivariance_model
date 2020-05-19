import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import torch.nn as nn
from matplotlib import colors
import matplotlib.pyplot as plt
from metric import *
from scipy.ndimage.interpolation import rotate as scipy_rotate


### CONSTANT
"""Pascal VOC Dataset Segmentation Dataloader"""
VOC_CLASSES = ('background',  # always index 0
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES) + 1
def get_voc_cst() -> (tuple,int):
    
    return VOC_CLASSES,NUM_CLASSES


### SAVE AND DATASETS UTILS FUNCTIONS

def save_loss(model_name):
    """
        Save losses into numpy files.
    """
    save = os.path.join(SAVE_DIR,model_name+'_loss_train.npy')
    np.save(save,np.array(loss_train))
    save = os.path.join(SAVE_DIR,model_name+'_loss_test.npy')
    np.save(save,np.array(loss_test))
    save = os.path.join(SAVE_DIR,model_name+'_iou_train.npy')
    np.save(save,np.array(iou_train))
    save = os.path.join(SAVE_DIR,model_name+'_iou_test.npy')
    np.save(save,np.array(iou_test))
    
    
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
    labeled, unlabeled = random_split(dataset, lengths)
    train_full_supervised = Split_Dataset(
        labeled)
    torch.manual_seed(torch.initial_seed())
    return train_full_supervised

### METRICS FUNCTIONS 

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
  with torch.no_grad():
    for i,(x,mask) in enumerate(val_loader):
          x = x.to(device)
          mask = mask.to(device)
          model.eval()
          pred = model(x)
          try:
                pred = pred["out"]
          except:
                print('')
            
          loss = criterion(pred,mask)
          loss_test.append(loss.item())
          s = scores(pred.max(dim=1)[1],mask)
          IoU = inter_over_union(pred.argmax(dim=1).detach().cpu(),mask.detach().cpu())
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
          if plot:
              plot_pred_mask(pred.argmax(dim=1).detach().cpu()[0],mask.detach().cpu()[0],cmap=None,iou=True)
           

    

    print("Mean IOU :",np.array(iou_test).mean(),"Pixel Accuracy :",np.array(pixel_accuracy).mean(),"Loss Validation :",np.array(loss_test).mean())

### EQUIVARIANCE UTILS FUNCTIONS 

# rotate images
def rotate_image(image,angle,reshape=False):
    """
        Rotate a tensor with a certain angle.
        If true, expands the output image to make it large enough to hold the entire rotated image.
        Else it keeps the same size
    """
    #image = image.squeeze()
    if len(image.size())==3: # Case of a single image.
        axes = ((1,2))
    elif len(image.size())==4: # Case of a batch of images
        axes = ((2,3))
    else:
        print("Size must be 4 or 5.")
        return 
    im = scipy_rotate(image.numpy(),angle=angle,reshape=reshape,axes=axes)
    im_t = torch.FloatTensor(im)
    return (im_t,360-angle)


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
    
def compute_transformations_batch(x,model,angle,reshape=False,criterion=nn.KLDivLoss(reduction='mean'),device='cpu'):
    """
    
    """
    x = x.to(device)
    rot_x,new_angle = rotate_image(x.detach().cpu(),angle=angle,reshape=reshape)
    softmax = nn.Softmax(dim=1)
    try:
        pred_x = model(x.to(device))['out'] # a prediction of the original images.
        pred_rot = model(rot_x.to(device))['out'] # a prediction of the rotated images.
    except:
        pred_x = model(x.to(device))
        pred_rot = model(rot_x.to(device))
    
    pred_droit = rotate_mask(pred_rot.detach().cpu(),new_angle,reshape=reshape)
    loss = criterion(softmax(pred_x.cpu()).log(),softmax(pred_droit.cpu()))
    acc = scores(pred_x.argmax(dim=1).detach().cpu(),pred_droit.argmax(dim=1).detach().cpu())["Pixel Accuracy"]
    # compare the pred on the original images and the pred on the rotated images put back in place
    return loss,acc  

## METRIC FUNCTIONS
def inter_over_union(pred, mask, num_class=21):
    """
        https://github.com/chenxi116/DeepLabv3.pytorch/blob/master/utils.py
        Inter over Union functions using numpy fast histogram.
        
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











### PLOT UTILS FUNCTIONS

def get_cmap() -> colors.ListedColormap:
    """
        return a cmap for pascal voc_dataset 
    """
    cmap_test = colors.ListedColormap(['black','green','blue','yellow','pink','orange','maroon','darkorange'\
                                 ,'skyblue','chocolate','azure','hotpink','tan','gold','silver','navy','white'\
                                ,'olive','beige','brown','royalblue','violet'])
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
    plt.imshow(mask,cmap=cmap,vmin=0,vmax=21) #plt.cm.get_cmap('cubehelix', 21)
    a = fig.add_subplot(1,2,2)
    plt.imshow(pred,cmap=cmap,vmin=0,vmax=NUM_CLASSES)
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
        
        
    