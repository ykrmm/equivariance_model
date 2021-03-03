import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.metrics import ConfusionMatrix, mIoU,Accuracy,Loss
import torch 
import torch.nn as nn
import numpy as np
import utils as U
import random 
import get_datasets as gd
from matplotlib import colors
import os


##############


###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                           FULLY SUPERVISED TRAINING
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|
def step_train_supervised(model,train_loader,criterion,optimizer,device='cpu',num_classes=21):
    """
        A step of fully supervised segmentation model training.
    """

    def train_function(engine, batch):
        optimizer.zero_grad()
        model.train()  
        img,mask = batch       
        img = img.to(device)
        mask = mask.to(device)   
        mask_pred = model(img)
        try:
            mask_pred = mask_pred['out'] 
        except:
            print('')   
        #print(mask_pred)
        #print("UNIQUE",torch.unique(mask_pred.argmax(dim=1)))
        #print("SIZE",mask_pred.size())
        loss = criterion(mask_pred, mask)
        loss.backward()
        optimizer.step()
        
        return mask_pred, mask
    #print(num_classes)
    
    train_engine = Engine(train_function)
    cm = ConfusionMatrix(num_classes=num_classes)#,output_transform=output_transform)
    mIoU(cm).attach(train_engine, 'mean IoU')   
    Accuracy().attach(train_engine, "accuracy")
    Loss(loss_fn=nn.CrossEntropyLoss()).attach(train_engine, "CE Loss")
    state = train_engine.run(train_loader)
    #print("mIoU :",state.metrics['mean IoU'])
    #print("Accuracy :",state.metrics['accuracy'])
    #print("CE Loss :",state.metrics['CE Loss'])
    
    return state
    
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
def train_fully_supervised(model,n_epochs,train_loader,val_loader,criterion,optimizer,scheduler,\
        save_folder,model_name,benchmark=False,save_all_ep=True, save_best=False, device='cpu',num_classes=21):
    """
        A complete training of fully supervised model. 
        save_folder : Path to save the model, the courb of losses,metric...
        benchmark : enable or disable backends.cudnn 
        save_all_ep : if True, the model is saved at each epoch in save_folder
        scheduler : if True, the model will apply a lr scheduler during training
    """
    torch.backends.cudnn.benchmark=benchmark
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader) * n_epochs)) ** 0.9)
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
        if scheduler:
            lr_scheduler.step()
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
        U.save_model(model,save_all_ep,save_best,save_folder,model_name,ep=ep,iou=iou,iou_test=iou_test)

    U.save_curves(path=save_folder,loss_train=loss_train,iou_train=iou_train,accuracy_train=accuracy_train\
                                ,loss_test=loss_test,iou_test=iou_test,accuracy_test=accuracy_test)


###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                               ROTATION EQUIV
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|

def train_step_rot_equiv(model,train_loader_sup,train_loader_equiv,criterion_supervised,criterion_unsupervised,\
                        optimizer,gamma,Loss,device,num_classes=21,angle_max=30):
    """
        A training epoch for rotational equivariance using for semantic segmentation
    """
    l_loss_equiv = []
    l_loss_sup = []
    l_loss = []
    equiv_acc = [] # Equivariance accuracy btwn the mask of the input rotated image and the mask of non rotated image
    model.train()
    for batch_sup,batch_unsup in zip(train_loader_sup,train_loader_equiv):
        optimizer.zero_grad()
        if random.random() > 0.5: # I use this to rotate the image on the left and on the right during training.
            angle = np.random.randint(0,angle_max)
        else:
            angle = np.random.randint(360-angle_max,360)
        x_unsup,_ = batch_unsup
        loss_equiv,acc = U.compute_transformations_batch(x_unsup,model,angle,reshape=False,\
                                                     criterion=criterion_unsupervised,Loss = Loss,\
                                                       device=device)
        x,mask = batch_sup
        x = x.to(device)
        mask = mask.to(device)
        pred = model(x)["out"]
        loss_equiv = loss_equiv.to(device) # otherwise bug in combining the loss 
        loss_sup = criterion_supervised(pred,mask)
        loss = gamma*loss_sup + (1-gamma)*loss_equiv # combine loss              
        loss.backward()
        optimizer.step()
        l_loss.append(float(loss))
        l_loss_equiv.append(float(loss_equiv))
        l_loss_sup.append(float(loss_sup))
        equiv_acc.append(acc)
    state = eval_model(model,train_loader_equiv,device=device,num_classes=num_classes)
    iou = state.metrics['mean IoU']
    accuracy = state.metrics['accuracy']
    d = {'loss':np.array(l_loss).mean(),'loss_equiv':np.array(l_loss_equiv).mean(),\
        'loss_sup':np.array(l_loss_sup).mean(),'equiv_acc':np.array(equiv_acc).mean(),'iou_train':iou,'accuracy_train':accuracy}
    return d

def train_rot_equiv(model,n_epochs,train_loader_sup,train_dataset_unsup,val_loader,criterion_supervised,optimizer,scheduler,\
        Loss,gamma,batch_size,save_folder,model_name,benchmark=False,angle_max=30,size_img=520,\
        eval_every=5,save_all_ep=True,dataroot_voc='~/data/voc2012',save_best=False, device='cpu',num_classes=21):
    """
        A complete training of rotation equivariance supervised model. 
        save_folder : Path to save the model, the courb of losses,metric...
        benchmark : enable or disable backends.cudnn 
        Loss : Loss for unsupervised training 'KL' 'CE' 'L1' or 'MSE'
        gamma : float btwn [0,1] -> Balancing two losses loss_sup*gamma + (1-gamma)*loss_unsup
        save_all_ep : if True, the model is saved at each epoch in save_folder
        scheduler : if True, the model will apply a lr scheduler during training
        eval_every : Eval Model with different input image angle every n step
        size_img : size of image during evaluation
        angle_max : max angle rotation for input images
    """
    torch.backends.cudnn.benchmark=benchmark
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader_sup) * n_epochs)) ** 0.9)
    criterion_unsupervised = U.get_criterion(Loss)
    iou_train = []
    iou_test = []
    combine_loss_train = []
    combine_loss_test = []
    loss_train_unsup = []
    loss_train_sup = []
    loss_test_unsup = []
    loss_test_sup = []
    equiv_accuracy_train = []
    equiv_accuracy_test = []
    accuracy_test = []
    accuracy_train = []
    for ep in range(n_epochs):
        train_loader_equiv = torch.utils.data.DataLoader(train_dataset_unsup,batch_size=batch_size,\
                                                     shuffle=True,drop_last=True)
        print("EPOCH",ep)
        # TRAINING
        d = train_step_rot_equiv(model,train_loader_sup,train_loader_equiv,criterion_supervised,criterion_unsupervised,\
                        optimizer,gamma,Loss,device,angle_max=angle_max,num_classes=num_classes)
        if scheduler:
            lr_scheduler.step()
        combine_loss_train.append(d['loss'])
        loss_train_unsup.append(d['loss_equiv'])
        loss_train_sup.append(d['loss_sup'])
        equiv_accuracy_train.append(d['equiv_acc'])
        iou_train.append(d['iou_train'])
        accuracy_train.append(d['accuracy_train'])
        print('TRAIN - EP:',ep,'iou:',d['iou_train'],'Accuracy:',d['accuracy_train'],'Loss sup:',d['loss_sup'],\
            'Loss equiv:',d['loss_equiv'],'Combine Loss:',d['loss'],'Equivariance Accuracy:',d['equiv_acc'],)
        # EVALUATION 
        model.eval()
        with torch.no_grad():
            state = eval_model(model,val_loader,device=device,num_classes=num_classes)
            iou = state.metrics['mean IoU']
            acc = state.metrics['accuracy']
            loss = state.metrics['CE Loss'] 
            loss_test_sup.append(loss)
            iou_test.append(iou)
            accuracy_test.append(acc)
            print('TEST - EP:',ep,'iou:',iou,'Accuracy:',acc,'Loss CE',loss)
            U.save_model(model,save_all_ep,save_best,save_folder,model_name,ep=ep,iou=iou,iou_test=iou_test)
            
            
            if ep%eval_every==0: # Eval loss equiv and equivariance accuracy for the validation dataset
                equiv_acc, m_loss_equiv = U.eval_accuracy_equiv(model,val_loader,criterion=criterion_unsupervised,\
                                nclass=21,device=device,Loss=Loss,plot=False,angle_max=angle_max,random_angle=False)
                loss_test_unsup.append(m_loss_equiv)
                equiv_accuracy_test.append(equiv_acc)  
                """  
                print('VOC Dataset Train')
                _ = eval_model_all_angle(model,size_img,dataroot_voc,train=True,device=device,num_classes=num_classes)
                print('VOC Dataset Val')
                _ = eval_model_all_angle(model,size_img,dataroot_voc,train=False,device=device,num_classes=num_classes)
                ## Save model"""
                

    U.save_curves(path=save_folder,combine_loss_train=combine_loss_train,loss_train_sup=loss_train_sup,\
    loss_train_unsup=loss_train_unsup,iou_train=iou_train,accuracy_train=accuracy_train,equiv_accuracy_train=equiv_accuracy_train,\
    combine_loss_test=combine_loss_test,loss_test_unsup=loss_test_unsup,equiv_accuracy_test=equiv_accuracy_test,\
    loss_test_sup= loss_test_sup,iou_test=iou_test,accuracy_test=accuracy_test)


def eval_model_all_angle(model,size,dataroot_voc,train=False,batch_size=1,device='cpu',num_classes=21):
    """
        Eval IoU with different angle in the input images.        
        train : Bool -> Use train dataset or not. 
    """
    l_angle = [0,10,20,30,330,340,350]
    d_iou = {}
    d_iou = d_iou.fromkeys(l_angle,None)
    for angle in l_angle:
        if not train:          
            dataloader = gd.get_dataset_val(batch_size,angle,size,dataroot_voc)
        else:
            dataloader = gd.get_dataset_train_VOC(batch_size,angle,size,dataroot_voc)
        state = eval_model(model,dataloader,device=device,num_classes=num_classes)
        d_iou[angle] = {'mIoU':state.metrics['mean IoU'],'Accuracy':state.metrics['accuracy'],\
            'CE Loss':state.metrics['CE Loss']}
    for k in d_iou.keys():
        print('Scores for datasets rotate by',k,'degrees:')
        print('   mIoU',d_iou[k]['mIoU'],'Accuracy',d_iou[k]['Accuracy'],'CE Loss',d_iou[k]['CE Loss'])
    return d_iou


###########################################################################################################################|
#--------------------------------------------------------------------------------------------------------------------------|
#                                               SCALE EQUIVARIANCE
#--------------------------------------------------------------------------------------------------------------------------|
###########################################################################################################################|
def train_step_scale_equiv(model,train_loader_sup,train_loader_equiv,criterion_supervised,criterion_unsupervised,\
                        optimizer,gamma,Loss,device,size_img,scale_factor=(0.5,1.2)):
    """
        A training epoch for rotational equivariance using for semantic segmentation
    """
    l_loss_equiv = []
    l_loss_sup = []
    l_loss = []
    equiv_acc = [] # Equivariance accuracy btwn the mask of the input rotated image and the mask of non rotated image
    model.train()
    for batch_sup,batch_unsup in zip(train_loader_sup,train_loader_equiv):
        optimizer.zero_grad()
        min_f,max_f = scale_factor
        factor = random.uniform(min_f,max_f) 
        size = int(size_img * factor)
        x_unsup,_ = batch_unsup
        loss_equiv,acc = U.compute_scale_equiv_batch(x_unsup,model,size=(size,size),\
                                                     criterion=criterion_unsupervised,Loss = Loss,\
                                                       device=device)
        x,mask = batch_sup
        x = x.to(device)
        mask = mask.to(device)
        pred = model(x)["out"]
        loss_equiv = loss_equiv.to(device) # otherwise bug in combining the loss 
        print('Loss equiv',loss_equiv)
        loss_sup = criterion_supervised(pred,mask)
        loss = gamma*loss_sup + (1-gamma)*loss_equiv # combine loss              
        loss.backward()
        optimizer.step()
        l_loss.append(float(loss))
        l_loss_equiv.append(float(loss_equiv))
        l_loss_sup.append(float(loss_sup))
        equiv_acc.append(acc)
    state = eval_model(model,train_loader_equiv,device=device,num_classes=21)
    iou = state.metrics['mean IoU']
    accuracy = state.metrics['accuracy']
    d = {'loss':np.array(l_loss).mean(),'loss_equiv':np.array(l_loss_equiv).mean(),\
        'loss_sup':np.array(l_loss_sup).mean(),'equiv_acc':np.array(equiv_acc).mean(),'iou_train':iou,'accuracy_train':accuracy}
    return d


def train_scale_equiv(model,n_epochs,train_loader_sup,train_dataset_unsup,val_loader,criterion_supervised,optimizer,scheduler,\
        Loss,gamma,batch_size,save_folder,model_name,benchmark=False,angle_max=30,size_img=520,scale_factor=(0.5,1.2),\
        save_all_ep=True,dataroot_voc='~/data/voc2012',save_best=False, device='cpu',num_classes=21):
    """
        A complete training of rotation equivariance supervised model. 
        save_folder : Path to save the model, the courb of losses,metric...
        benchmark : enable or disable backends.cudnn 
        Loss : Loss for unsupervised training 'KL' 'CE' 'L1' or 'MSE'
        gamma : float btwn [0,1] -> Balancing two losses loss_sup*gamma + (1-gamma)*loss_unsup
        save_all_ep : if True, the model is saved at each epoch in save_folder
        scheduler : if True, the model will apply a lr scheduler during training
        eval_every : Eval Model with different input image angle every n step
        size_img : size of image during evaluation
        scale_factor : scale between min*size_img and max*size_img
    """
    torch.backends.cudnn.benchmark=benchmark
    if scheduler:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader_sup) * n_epochs)) ** 0.9)
    criterion_unsupervised = U.get_criterion(Loss)
    print('Criterion Unsupervised',criterion_unsupervised)
    iou_train = []
    iou_test = []
    combine_loss_train = []
    combine_loss_test = []
    loss_train_unsup = []
    loss_train_sup = []
    loss_test_unsup = []
    loss_test_sup = []
    equiv_accuracy_train = []
    equiv_accuracy_test = []
    accuracy_test = []
    accuracy_train = []
    torch.autograd.set_detect_anomaly(True)
    for ep in range(n_epochs):
        train_loader_equiv = torch.utils.data.DataLoader(train_dataset_unsup,batch_size=batch_size,\
                                                     shuffle=True,drop_last=True)
        print("EPOCH",ep)
        # TRAINING
        d = train_step_scale_equiv(model,train_loader_sup,train_loader_equiv,criterion_supervised,criterion_unsupervised,\
                        optimizer,gamma,Loss,device,size_img=size_img,scale_factor=scale_factor)
        if scheduler:
            lr_scheduler.step()
        combine_loss_train.append(d['loss'])
        loss_train_unsup.append(d['loss_equiv'])
        loss_train_sup.append(d['loss_sup'])
        equiv_accuracy_train.append(d['equiv_acc'])
        iou_train.append(d['iou_train'])
        accuracy_train.append(d['accuracy_train'])
        print('TRAIN - EP:',ep,'iou:',d['iou_train'],'Accuracy:',d['accuracy_train'],'Loss sup:',d['loss_sup'],\
            'Loss equiv:',d['loss_equiv'],'Combine Loss:',d['loss'],'Equivariance Accuracy:',d['equiv_acc'],)
        # EVALUATION 
        model.eval()
        with torch.no_grad():
            state = eval_model(model,val_loader,device=device,num_classes=num_classes)
            iou = state.metrics['mean IoU']
            acc = state.metrics['accuracy']
            loss = state.metrics['CE Loss'] 
            loss_test_sup.append(loss)
            iou_test.append(iou)
            accuracy_test.append(acc)
            print('TEST - EP:',ep,'iou:',iou,'Accuracy:',acc,'Loss CE',loss)

    U.save_curves(path=save_folder,combine_loss_train=combine_loss_train,loss_train_sup=loss_train_sup,\
    loss_train_unsup=loss_train_unsup,iou_train=iou_train,accuracy_train=accuracy_train,equiv_accuracy_train=equiv_accuracy_train,\
    combine_loss_test=combine_loss_test,loss_test_unsup=loss_test_unsup,equiv_accuracy_test=equiv_accuracy_test,\
    loss_test_sup= loss_test_sup,iou_test=iou_test,accuracy_test=accuracy_test)