from sklearn.metrics import confusion_matrix
import torch
import torchvision
import numpy as np
cm =  confusion_matrix(gt_r,t_r,labels=[0,1,2,3])
iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag() + 1e-15)


class IoU(object):
    def __init__(self,n_classes=4) -> None:
        self.cm = [] # aji metrics
        self.first = True 
        self.labels = np.arange(n_classes)
        self.iou = []

    def add_prediction(self,labels_pred,gt):
        """
            Input labels_pred -> (1,C,H,W) torch.Tensor BS 1 for the moment
            GT -> torch.Tensor Ground truth (1,H,W)
        """
        labels_pred = labels_pred.squeeze()
        gt = gt.squeeze()
        labels_pred = labels_pred.argmax(dim=0)
        labels_pred = labels_pred.reshape(labels_pred.size()[0]*labels_pred.size()[1])
        gt = gt.reshape(gt.size()[0]*gt.size()[1])
        gt = gt.long().detach().cpu().numpy()
        labels_pred = labels_pred.long().detach().cpu().numpy()

        if self.first:
            self.cm = confusion_matrix(gt,labels_pred)
            self.first = False
        else:
            self.cm += confusion_matrix(gt,labels_pred)

        

    

    def get_mIoU(self):
        
        return self.iou.mean()  # Average of IoU
    

    def get_IoU(self):
        self.iou = self.cm.diag() / (self.cm.sum(dim=1) + self.cm.sum(dim=0) - self.cm.diag() + 1e-15)
        return self.iou