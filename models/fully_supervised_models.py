import torch
from torchvision import models
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import functional as F 
import sys
sys.path.insert(1, '../utils')
sys.path.insert(1, '../datasets')
import my_datasets as mdset
from argparse import ArgumentParser
import torch.utils.data as tud


class Fully_Supervised_SegNet(pl.LightningModule):

    def __init__(self,model_name='FCN',pretrained=False,learning_rate = 10e-4,moment=0.9,wd=2e-4):
        """
            model type FCN or DLV3 
            pretrained : load a pretrained model of pytorch if true 
        """
        super().__init__()
        if model_name.upper()=='FCN':
            self.segnet = models.segmentation.fcn_resnet101(pretrained=pretrained)
        elif model_name.upper()=='DLV3':
            self.segnet = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        else:
            raise Exception('model must be "FCN" or "DLV3"')
        self.save_hyperparameters()
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        #self.train_cm = pl.metrics.classification.ConfusionMatrix(num_classes=21)
    def training_step(self, batch, batch_idx):
        x, mask = batch
        pred = self.segnet(x)["out"]
        loss = F.cross_entropy(pred, mask)
        self.train_acc(pred, mask)
        self.log('train_loss', loss)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        pred = self.segnet(x)["out"]
        loss = F.cross_entropy(pred, mask)
        self.val_acc(pred, mask)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        return loss
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,\
            momentum=self.hparams.moment,weight_decay=self.hparams.wd)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=10e-4)
        parser.add_argument('--wd', type=float, default=2e-4)
        parser.add_argument('--moment', type=float, default=0.9)
        return parser

def main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--model_name', default='FCN', type=str)
    parser.add_argument('--pretrained', default=False, type=bool,help="Use pretrained pytorch model")
    parser.add_argument('--rotate', default=False, type=bool,help="Use random rotation as data augmentation")
    parser.add_argument('--nw', default=0, type=int,help="Num workers for the data loader")
    parser.add_argument('--pm', default=True, type=bool,help="Pin memory for the dataloader")
    parser.add_argument('--dataroot_voc', default='/data/voc2012', type=str)
    parser.add_argument('--dataroot_sbd', default='/data/sbd', type=str)
    parser.add_argument('--auto_lr', default=False,type=bool,help="Call Pytorch lightning to find the best value for the lr")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Fully_Supervised_SegNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    train_dataset_VOC = mdset.VOCSegmentation(args.dataroot_voc,year='2012', image_set='train', download=True,rotate=args.rotate)
    val_dataset_VOC = mdset.VOCSegmentation(args.dataroot_voc,year='2012', image_set='val', download=True)
    train_dataset_SBD = mdset.SBDataset(args.dataroot_sbd, image_set='train_noval',mode='segmentation',rotate=args.rotate)#,\
        #download=True)
    train_dataset = tud.ConcatDataset([train_dataset_VOC,train_dataset_SBD])
    dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=args.nw,\
        pin_memory=args.pm,shuffle=True,drop_last=True)
    dataloader_val = torch.utils.data.DataLoader(val_dataset_VOC,num_workers=args.nw,pin_memory=args.pm,\
        batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = Fully_Supervised_SegNet(model_name=args.model_name, pretrained=args.pretrained,\
        learning_rate = args.learning_rate,moment=args.moment,wd=args.wd)

    # ------------
    # training
    # ------------
    if args.auto_lr==True: 
        trainer = pl.Trainer(gpus=1,auto_lr_find=True,max_epochs=args.n_epochs)
        trainer.tune(model)
        model.learning_rate
    else:
        trainer = pl.Trainer(gpus=1,max_epochs=args.n_epochs)
        
    trainer.fit(model, dataloader_train, dataloader_val)
    # ------------
    # testing
    # ------------
    #trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    main()