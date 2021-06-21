import argparse
from argparse import ArgumentParser
from os.path import isfile,join
import torch
import sys
sys.path.insert(1, 'utils')
sys.path.insert(1, 'datasets')
import my_datasets as mdset
import eval_train as ev 

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

def main():
    #torch.manual_seed(42)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    
    # Model and eval
    parser.add_argument('--model_name', default='rot_equiv_lc.pt', type=str,help="Model name")
    parser.add_argument('--model_dir', default='/share/homes/karmimy/equiv/save_model/rot_equiv_lc', type=str,help="Model name")
    parser.add_argument('--expe', default='17', type=str,help="Expe")
    args = parser.parse_args()

    # DATASETS
    dataroot_landcover = '/share/DEEPLEARNING/datasets/landcover'

    model_dir = args.model_dir # Saved model dir
    expe = args.expe
    model_name = args.model_name
    folder_model = join(model_dir,expe) 

    nw = 4
    pm = True
    # GPU 
    gpu = 0
    # EVAL PARAMETERS
    bs = 1  

    # DEVICE
    # Decide which device we want to run on
    device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
    print("device :",device)

    model = torch.load(join(folder_model,model_name),map_location=device)

    test_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="test")

    l_angles = [210,240,270,300,330,0,30,60,90,120,150]
    #l_angles = [330,340,350,0,10,20,30]
    l_iou = []
    l_iou_bg = []
    l_iou_c1  = []
    l_iou_c2 = []
    l_iou_c3 = []
    for angle in l_angles:
        test_dataset = mdset.LandscapeDataset(dataroot_landcover,image_set="test",fixing_rotate=True,angle_fix=angle)
        dataloader_val = torch.utils.data.DataLoader(test_dataset,num_workers=nw,pin_memory=pm,\
            batch_size=bs)
        if angle % 90 ==0:
            state = ev.eval_model(model,dataloader_val,device=device,num_classes=4)
            m_iou,iou = state.metrics['mean IoU'],state.metrics['IoU']
        else:
            m_iou,iou = ev.eval_model_tmetrics(model,dataloader_val,device=device,num_classes=4)

        
        try: 
            m_iou = m_iou.item()
        except:
            print(m_iou,type(m_iou))
            m_iou = float(m_iou)
        l_iou.append(round(m_iou,3))
        print('EVAL FOR ANGLE',angle,': IoU',m_iou)
        print('IoU All classes',iou)
        l_iou_bg.append(float(iou[0]))
        l_iou_c1.append(float(iou[1]))
        l_iou_c2.append(float(iou[2]))
        l_iou_c3.append(float(iou[3]))
    l_iou.append(l_iou[0])
    l_iou.append(l_iou_bg[0])
    l_iou.append(l_iou_c1[0])
    l_iou.append(l_iou_c2[0])
    l_iou.append(l_iou_c3[0])

    print('L_IOU',l_iou)
    print('L_IOU',l_iou_bg)
    print('L_IOU',l_iou_c1)
    print('L_IOU',l_iou_c2)
    print('L_IOU',l_iou_c3)



if __name__ == '__main__':
    main()
