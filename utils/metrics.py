from sklearn.metrics import confusion_matrix
import torch
from torch.functional import norm
import torchvision
import numpy as np



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

        #print(np.shape(labels_pred),np.shape(gt))
        #print(np.un)
        if self.first:
            self.cm = confusion_matrix(gt,labels_pred,labels=self.labels)
            #print(self.cm)
            #print('SHAPE first CM',np.shape(self.cm))
            self.first = False
        else:
            new_cm = confusion_matrix(gt,labels_pred,labels=self.labels)
            #print('SHAPE new CM',np.shape(new_cm))
            self.cm += new_cm

        

    

    def get_mIoU(self):
        
        return self.iou.mean()  # Average of IoU
    

    def get_IoU(self):
        self.cm = torch.Tensor(self.cm) # To use diag
        self.iou = self.cm.diag() / (self.cm.sum(dim=1) + self.cm.sum(dim=0) - self.cm.diag() + 1e-15)
        return self.iou