import torch
import numpy as np
from utils import * 



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
 
    



    

