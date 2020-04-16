import torch
import numpy as np

def save_loss(model_name):
    save = os.path.join(SAVE_DIR,model_name+'_loss_train.npy')
    np.save(save,np.array(loss_train))
    save = os.path.join(SAVE_DIR,model_name+'_loss_test.npy')
    np.save(save,np.array(loss_test))
    save = os.path.join(SAVE_DIR,model_name+'_iou_train.npy')
    np.save(save,np.array(iou_train))
    save = os.path.join(SAVE_DIR,model_name+'_iou_test.npy')
    np.save(save,np.array(iou_test))