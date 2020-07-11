import torch
import numpy as np
import torchvision.transforms.functional as TF


def to_tensor_target(img):
  img = np.array(img)
  # border
  img[img==255] = 0 # border = background 
  return torch.LongTensor(img)

def rotate_pil(img,angle,fill=0):
    img = TF.rotate(img,angle=angle,fill=fill)
    return img

def hflip(img):
    img = TF.hflip(img)
    return img
