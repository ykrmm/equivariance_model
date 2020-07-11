from ignite.metrics import ConfusionMatrix, mIoU,Accuracy,Loss
from ignite.engine.engine import Engine
import torch 
import torch.nn as nn
import numpy as np
from utils import * 
from matplotlib import colors

CMAP  = colors.ListedColormap(['black','green','blue','yellow','pink','orange','maroon','darkorange'\
                                 ,'skyblue','chocolate','azure','hotpink','tan','gold','silver','navy','white'\
                                ,'olive','beige','brown','royalblue','violet'])


def eval_model(model,val_loader,device='cpu',num_classes=21):

    def evaluate_function(engine, batch):
        model.eval()
        with torch.no_grad():
            img, mask = batch
            img = img.to(device)
            mask = mask.to(device)
            mask_pred = model(img)
            try:
                mask_pred = mask_pred['out'] 
            except:
                print('')
            return mask_pred, mask

    val_evaluator = Engine(evaluate_function)
    cm = ConfusionMatrix(num_classes=num_classes)
    mIoU(cm).attach(val_evaluator, 'mean IoU')   
    Accuracy().attach(val_evaluator, "accuracy")
    Loss(loss_fn=nn.CrossEntropyLoss())\
    .attach(val_evaluator, "CE Loss")

    state = val_evaluator.run(val_loader)
    #print("mIoU :",state.metrics['mean IoU'])
    #print("Accuracy :",state.metrics['accuracy'])
    #print("CE Loss :",state.metrics['CE Loss'])
    
    return state

def train_model_supervised(model,train_loader,criterion,optimizer,device='cpu',num_classes=21):


    def train_function(engine, batch):
        model.train()       
        img, mask = batch[0].to(device), batch[1].to(device)       
        mask_pred = model(img)
        try:
            mask_pred = mask_pred['out'] 
        except:
            print('')
        loss = criterion(mask_pred, mask)
        loss.backward()
        optimizer.step()
        return loss.item()


    train_engine = Engine(train_function)
    cm = ConfusionMatrix(num_classes=num_classes)
    mIoU(cm).attach(train_engine, 'mean IoU')   
    Accuracy().attach(train_engine, "accuracy")
    Loss(loss_fn=nn.CrossEntropyLoss())\
    .attach(train_engine, "CE Loss")

    state = train_engine.run(train_loader)
    #print("mIoU :",state.metrics['mean IoU'])
    #print("Accuracy :",state.metrics['accuracy'])
    #print("CE Loss :",state.metrics['CE Loss'])
    
    return state
    
def eval_model_all_angle(model,batch_size=1,device='cpu',num_classes=21):
    l_angle = [10,20,30,330,340,350]
    d_iou = {}
    d_iou = d_iou.fromkeys(l_angle,None)
    for angle in l_angle:
        val_loader = get_dataset_val(batch_size,angle)
        state = eval_model(model,val_loader,device=device,num_classes=num_classes)
        d_iou[angle] = {'mIoU':state.metrics['mean IoU'],'Accuracy':state.metrics['accuracy'],\
            'CE Loss':state.metrics['CE Loss']}
    
    return d_iou



