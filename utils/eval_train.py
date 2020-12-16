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
import get_datasets as gd
from matplotlib import colors
import os
"""
def create_trainer(model, optimizer, criterion, lr_scheduler, config):

    def train_step(engine, batch):
        x, y = batch[0].to(idist.device()), batch[1].to(idist.device())
        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        return loss.item()

    # Define trainer engine
    trainer = Engine(train_step)

    if idist.get_rank() == 0:
        # Add any custom handlers
        @trainer.on(Events.ITERATION_COMPLETED(every=200))
        def save_checkpoint():
            fp = Path(config.get("output_path", "output")) / "checkpoint.pt"
            torch.save(model.state_dict(), fp)

        # Add progress bar showing batch loss value
        ProgressBar().attach(trainer, output_transform=lambda x: {"batch loss": x})

    return trainer


# slide 2 ####################################################################


def training(local_rank, config):

    # Setup dataflow and
    train_loader, val_loader = get_dataflow(config)
    model, optimizer, criterion, lr_scheduler = initialize(config)

    # Setup model trainer and evaluator
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, config)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()}, device=idist.device())

    # Run model evaluation every 3 epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=3))
    def evaluate_model():
        state = evaluator.run(val_loader)
        if idist.get_rank() == 0:
            print(state.metrics)

    # Setup tensorboard experiment tracking
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            config.get("output_path", "output"), trainer, optimizer, evaluators={"validation": evaluator},
        )

    trainer.run(train_loader, max_epochs=config.get("max_epochs", 3))

    if idist.get_rank() == 0:
        tb_logger.close()


"""

##############

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

    def train_function(engine, batch):
        optimizer.zero_grad()
        model.train()       
        img, mask = batch   
        img = img.to(device)
        mask = mask.to(device)    
        mask_pred = model(img)
        try:
            mask_pred = mask_pred['out'] 
        except:
            print('')
        loss = criterion(mask_pred, mask)
        loss.backward()
        optimizer.step()
        return mask_pred, mask
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


def eval_model_all_angle(model,size,train=False,batch_size=1,device='cpu',num_classes=21):
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



