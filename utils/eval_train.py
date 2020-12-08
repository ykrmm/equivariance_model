from ignite.metrics import ConfusionMatrix, mIoU,Accuracy,Loss
from ignite.engine.engine import Engine
import torch 
import torch.nn as nn
import numpy as np
import utils as U
import get_datasets as gd
from matplotlib import colors
import os


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

def step_train_supervised(model,train_loader,criterion,optimizer,device='cpu',num_classes=21):
    """
        A step of fully supervised segmentation model training.
    """
    def output_transform(output):
        output = output[1],output[2]
        return output

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
        return loss.item(),mask_pred, mask
    train_engine = Engine(train_function)
    cm = ConfusionMatrix(num_classes=num_classes,output_transform=output_transform)
    mIoU(cm).attach(train_engine, 'mean IoU')   
    Accuracy(output_transform=output_transform).attach(train_engine, "accuracy")
    Loss(loss_fn=nn.CrossEntropyLoss(),output_transform=output_transform).attach(train_engine, "CE Loss")
    state = train_engine.run(train_loader)
    #print("mIoU :",state.metrics['mean IoU'])
    #print("Accuracy :",state.metrics['accuracy'])
    #print("CE Loss :",state.metrics['CE Loss'])
    
    return state

def train_fully_supervised(model,n_epochs,train_loader,val_loader,criterion,optimizer,save_folder,model_name,benchmark=False,save_all_ep=True,\
                        device='cpu',num_classes=21):
    """
        A complete training of fully supervised model. 
        save_folder : Path to save the model, the courb of losses,metric...
        benchmark : enable or disable backends.cudnn 
        save_all_ep : if True, the model is saved at each epoch in save_folder
    """
    torch.backends.cudnn.benchmark=benchmark
    loss_test = []
    loss_train = []
    iou_train = []
    iou_test = []
    accuracy_train = []
    accuracy_test = []
    for ep in range(n_epochs):
        print("EPOCH",ep)
        model.train()
        state = step_train_supervised(model,train_loader=train_loader,criterion=criterion,\
            optimizer=optimizer,device=device,num_classes=num_classes)
        iou = state.metrics['mean IoU']
        acc = state.metrics['accuracy']
        loss = state.metrics['CE Loss'] 
        loss_train.append(loss)
        iou_train.append(iou)
        accuracy_train.append(acc)
        print('TRAIN - EP:',ep,'iou:',iou,'Accuracy:',acc,'Loss CE',loss)

        #Eval model
        model.eval()
        with torch.no_grad():
            state = eval_model(model,val_loader,device=device,num_classes=num_classes)
            iou = state.metrics['mean IoU']
            acc = state.metrics['accuracy']
            loss = state.metrics['CE Loss'] 
            loss_test.append(loss)
            iou_test.append(iou)
            accuracy_test.append(acc)
            print('TEST - EP:',ep,'iou:',iou,'Accuracy:',acc,'Loss CE',loss)
        
        ## Save model
        if save_all_ep:
            save_model = model_name+'_ep'+str(ep)+'.pt'
            save = os.path.join(save_folder,save_model)
            torch.save(model,save)
        else:
            save_model = model_name+'.pt'
            save = os.path.join(save_folder,save_model)
            torch.save(model,save)

    U.save_curves(path=save_folder,loss_train=loss_train,iou_train=iou_train,accuracy_train=accuracy_train\
                                ,loss_test=loss_test,iou_test=iou_test,accuracy_test=accuracy_test)


def eval_model_all_angle(model,train=False,batch_size=1,device='cpu',num_classes=21):
    """
        Eval IoU with different angle in the input images.        
        train : Bool -> Use train dataset or not. 
    """
    l_angle = [0,10,20,30,330,340,350]
    d_iou = {}
    d_iou = d_iou.fromkeys(l_angle,None)
    for angle in l_angle:
        if not train:          
            dataloader = gd.get_dataset_val(batch_size,angle)
        else:
            dataloader = gd.get_dataset_train_VOC(batch_size,angle)
        state = eval_model(model,dataloader,device=device,num_classes=num_classes)
        d_iou[angle] = {'mIoU':state.metrics['mean IoU'],'Accuracy':state.metrics['accuracy'],\
            'CE Loss':state.metrics['CE Loss']}
    for k in d_iou.keys():
        print('Scores for datasets rotate by',k,'degrees:')
        print('   mIoU',d_iou[k]['mIoU'],'Accuracy',d_iou[k]['Accuracy'],'CE Loss',d_iou[k]['CE Loss'])
    
    return d_iou



