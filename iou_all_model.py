import numpy as np
import os 
from os.path import join 



folder = '/share/homes/karmimy/equiv/save_model/rot_equiv_lc'


l_expe = [x[0] for x in os.walk(folder)]

for expe in l_expe:
    print('-------- EXPERIENCE --------')
    print(expe)

    print('-------- PARAMETERS ---------')
    try:
        with open(join(expe,'hparam.txt'), 'r') as f:
            print(f.read())

    except:
        print('NO FILE hparam.txt IN THIS EXPE')
    print('-------- PERFORMANCE --------- ')
    try:
        iou_max = np.max(np.load(join(expe,'iou_test.npy')))
        print('-----> mIoU TEST =',iou_max)
    except:
        print('NO IOU FOUND IN THIS EXPE')
    print('****************************************')
    print('\n')
    print('\n')
    print('****************************************')
    print('\n')

print('--------END -------')

