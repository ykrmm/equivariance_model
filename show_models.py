import argparse
from argparse import ArgumentParser
from os.path import isfile,join
import torch
import sys
sys.path.insert(1, 'utils')
sys.path.insert(1, 'datasets')
import os 
import numpy as np
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
    parser.add_argument('--model_dir', default='/share/homes/karmimy/equiv/save_model/rot_equiv', type=str,help="Model name")

    args = parser.parse_args()

    list_folder = [f[0] for f in os.walk(args.model_dir)][1:] 

    for d in list_folder:
        param_file = join(d,'hparam.txt')
        with open(param_file,'r') as fi:
            param = fi.read()
        print('------------------------ FOLDER: ',d)
        print('PARAMETERS')
        print('\n')
        print('.....')
        print(param)
        try:
            iou = np.load(join(d,'iou_test.npy'))
            print('mIoU MAX:',np.max(iou))

        except:
            print('No IoU found in',d)

        print('\n')
        print('\n')
        print('\n')
        print('.....')
        print('.....')
        

if __name__ == '__main__':
    main()