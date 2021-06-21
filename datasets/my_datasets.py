import os
import os.path
import tarfile
import collections
import torch
from torchvision.datasets.utils import download_url, check_integrity, verify_str_arg
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import PIL
from PIL import Image
import random
import shutil
import numpy as np
from typing import Any, Callable, Optional, Tuple



DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': os.path.join('VOCdevkit', 'VOC2012')
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': os.path.join('TrainVal', 'VOCdevkit', 'VOC2011')
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': os.path.join('VOCdevkit', 'VOC2010')
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': os.path.join('VOCdevkit', 'VOC2009')
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': os.path.join('VOCdevkit', 'VOC2008')
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    },
    '2007-test': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        'filename': 'VOCtest_06-Nov-2007.tar',
        'md5': 'b6e924de25625d8de591ea690078ad9f',
        'base_dir': os.path.join('VOCdevkit', 'VOC2007')
    }
}

class CocoDetection(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ) -> None:
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self) -> int:
        return len(self.ids)

class VOCSegmentation(VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.

    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 download=False,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 size_img=(520,520),
                 size_crop=(480,480),
                 scale_factor = (0.5,1.2),
                 p=0.5,
                 p_rotate = 0.25,
                 rotate=False,
                 scale=True):
        super(VOCSegmentation, self).__init__(root, transforms, transform, target_transform)
        ## Transformations
        self.mean = mean
        self.std = std 
        self.size_img = size_img
        self.size_crop = size_crop
        self.scale_factor = scale_factor
        self.p = p
        self.p_rotate = p_rotate
        self.rotate = rotate
        self.scale = scale
        self.train = image_set == 'train' or image_set == 'trainval'
        ##
        self.year = year
        if year == "2007" and image_set == "test":
            year = "2007-test"
        self.url = DATASET_YEAR_DICT[year]['url']
        self.filename = DATASET_YEAR_DICT[year]['filename']
        self.md5 = DATASET_YEAR_DICT[year]['md5']
        valid_sets = ["train", "trainval", "val"]
        if year == "2007-test":
            valid_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_sets)
        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    
    def my_transform(self, image, mask):
        # Resize
        
        if self.train and self.scale:
            min_size = int(self.size_img[0]*self.scale_factor[0])
            max_size = int(self.size_img[0]*self.scale_factor[1])
            if  min_size < self.size_crop[0]: 
                size = random.randint(self.size_crop[0],max_size)
            else:
                size = random.randint(min_size,max_size)
            resize = T.Resize((size,size))
        else:
            resize = T.Resize(self.size_img)
        image = resize(image)
        mask = resize(mask)

        if self.train : 
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=self.size_crop)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > self.p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if self.rotate:
                if random.random() > self.p_rotate:
                    if random.random() > 0.5:
                        angle = np.random.randint(0,30)
                        image = TF.rotate(image,angle=angle)
                        mask = TF.rotate(mask,angle=angle)
                    else:
                        angle = np.random.randint(330,360)
                        image = TF.rotate(image,angle=angle)
                        mask = TF.rotate(mask,angle=angle)
                    
        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image,self.mean,self.std)
        mask = to_tensor_target(mask)
        return image, mask

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        img, target = self.my_transform(img, target)
        return img, target


    def __len__(self):
        return len(self.images)




def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

def to_tensor_target(mask):
    mask = np.array(mask)
    # border
    mask[mask==255] = 0 # border = background 
    return torch.LongTensor(mask)

def to_tensor_target_lc(mask):
    # For the landcoverdataset
    mask = np.array(mask)
    mask = np.mean(mask, axis=2) 

    return torch.LongTensor(mask)




class SBDataset(VisionDataset):
    """`Semantic Boundaries Dataset <http://home.bharathh.info/pubs/codes/SBD/download.html>`_

    The SBD currently contains annotations from 11355 images taken from the PASCAL VOC 2011 dataset.

    .. note ::

        Please note that the train and val splits included with this dataset are different from
        the splits in the PASCAL VOC dataset. In particular some "train" images might be part of
        VOC2012 val.
        If you are interested in testing on VOC 2012 val, then use `image_set='train_noval'`,
        which excludes all val images.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of the Semantic Boundaries Dataset
        image_set (string, optional): Select the image_set to use, ``train``, ``val`` or ``train_noval``.
            Image set ``train_noval`` excludes VOC 2012 val images.
        mode (string, optional): Select target type. Possible values 'boundaries' or 'segmentation'.
            In case of 'boundaries', the target is an array of shape `[num_classes, H, W]`,
            where `num_classes=20`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version. Input sample is PIL image and target is a numpy array
            if `mode='boundaries'` or PIL image if `mode='segmentation'`.
    """

    url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
    md5 = "82b4d87ceb2ed10f6038a1cba92111cb"
    filename = "benchmark.tgz"

    voc_train_url = "http://home.bharathh.info/pubs/codes/SBD/train_noval.txt"
    voc_split_filename = "train_noval.txt"
    voc_split_md5 = "79bff800c5f0b1ec6b21080a3c066722"

    def __init__(self,
                 root,
                 image_set='train',
                 mode='segmentation',
                 download=False,
                 transforms=None,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 size_img=(520,520),
                 size_crop=(480,480),
                 scale_factor = (0.5,1.2),
                 p=0.5,
                 p_rotate=0.25,
                 rotate=False,
                 scale=True):

        try:
            from scipy.io import loadmat
            self._loadmat = loadmat
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: "
                               "pip install scipy")

        super(SBDataset, self).__init__(root, transforms)
        ## Transform
        self.mean = mean
        self.std = std 
        self.size_img = size_img
        self.size_crop = size_crop
        self.scale_factor = scale_factor
        self.p = p
        self.p_rotate = p_rotate
        self.rotate = rotate
        self.scale = scale
        self.train = image_set == 'train' or image_set == 'train_noval'
        ##
        self.image_set = verify_str_arg(image_set, "image_set",
                                        ("train", "val", "train_noval"))
        self.mode = verify_str_arg(mode, "mode", ("segmentation", "boundaries"))
        self.num_classes = 20

        sbd_root = self.root
        image_dir = os.path.join(sbd_root, 'img')
        mask_dir = os.path.join(sbd_root, 'cls')

        if download:
            download_extract(self.url, self.root, self.filename, self.md5)
            extracted_ds_root = os.path.join(self.root, "benchmark_RELEASE", "dataset")
            for f in ["cls", "img", "inst", "train.txt", "val.txt"]:
                old_path = os.path.join(extracted_ds_root, f)
                shutil.move(old_path, sbd_root)
            download_url(self.voc_train_url, sbd_root, self.voc_split_filename,
                         self.voc_split_md5)

        if not os.path.isdir(sbd_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        split_f = os.path.join(sbd_root, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".mat") for x in file_names]
        assert (len(self.images) == len(self.masks))

        self._get_target = self._get_segmentation_target \
            if self.mode == "segmentation" else self._get_boundaries_target

    def _get_segmentation_target(self, filepath):
        mat = self._loadmat(filepath)
        return Image.fromarray(mat['GTcls'][0]['Segmentation'][0])

    def _get_boundaries_target(self, filepath):
        mat = self._loadmat(filepath)
        return np.concatenate([np.expand_dims(mat['GTcls'][0]['Boundaries'][0][i][0].toarray(), axis=0)
                               for i in range(self.num_classes)], axis=0)

    def my_transform(self, image, mask):
        # Resize     
        if self.train and self.scale:
            min_size = int(self.size_img[0]*self.scale_factor[0])
            max_size = int(self.size_img[0]*self.scale_factor[1])
            if  min_size < self.size_crop[0]: 
                size = random.randint(self.size_crop[0],max_size)
            else:
                size = random.randint(min_size,max_size)
            resize = T.Resize((size,size))
        else:
            resize = T.Resize(self.size_img)
        image = resize(image)
        mask = resize(mask)

        if self.train : 
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=self.size_crop)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > self.p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            if self.rotate:
                if random.random() > self.p_rotate:
                    if random.random() > 0.5:
                        angle = np.random.randint(0,30)
                        image = TF.rotate(image,angle=angle)
                        mask = TF.rotate(mask,angle=angle)
                    else:
                        angle = np.random.randint(330,360)
                        image = TF.rotate(image,angle=angle)
                        mask = TF.rotate(mask,angle=angle)

        

        # Transform to tensor
        image = TF.to_tensor(image)
        image = TF.normalize(image,self.mean,self.std)
        mask = to_tensor_target(mask)
        return image, mask

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self._get_target(self.masks[index])
        img, target = self.my_transform(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Image set: {image_set}", "Mode: {mode}"]
        return '\n'.join(lines).format(**self.__dict__)


###########################################################################################
###########################################################################################
#                                                                                         #
#                                      LANDSCAPE                                          #
#                                                                                         #
###########################################################################################
###########################################################################################

class LandscapeDataset(Dataset):
    def __init__(self,
                 dataroot,
                 image_set='trainval',
                 mean = [0.485, 0.456, 0.406],
                 std = [0.229, 0.224, 0.225],
                 size_img = (512,512),
                 size_crop = (480,480),
                 scale_factor = (0.5,1.2),
                 p = 0.5,
                 p_rotate = 0.25,
                 rotate = False,
                 scale = True,
                 normalize = True,
                 pi_rotate = True,
                 fixing_rotate = False,
                 angle_fix = 0,
                 angle_max = 360):
        super(LandscapeDataset).__init__()

        ## Transform
        self.mean = mean
        self.std = std 
        self.size_img = size_img
        self.size_crop = size_crop
        self.scale_factor = scale_factor
        self.p = p
        self.p_rotate = p_rotate
        self.rotate = rotate
        self.scale = scale
        self.normalize = normalize # Use un-normalize image for plotting
        self.pi_rotate = pi_rotate # Use only rotation 90,180,270 rotations
        self.fixing_rotate = fixing_rotate # For test time, allow to eval a model on the dataset with a certain angle
        self.angle_fix = angle_fix # The angle used for the fixing rotation
        self.angle_max = angle_max

        if fixing_rotate: 
            self.rotate = False

        ##

        if image_set!= 'train' and image_set!='trainval' and image_set!='test':
            raise Exception("image set should be 'train' 'trainval' or 'test',not",image_set)
        self.train = image_set == 'train' or image_set == 'trainval'
        self.root_img = os.path.join(dataroot,'output')
        split_f = os.path.join(dataroot, image_set.rstrip('\n') + '.txt')
        with open(os.path.join(split_f), "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

    def my_transform(self, image, mask):
        # Resize     
        if self.train and self.scale:
            min_size = int(self.size_img[0]*self.scale_factor[0])
            max_size = int(self.size_img[0]*self.scale_factor[1])
            if  min_size < self.size_crop[0]: 
                size = random.randint(self.size_crop[0],max_size)
            else:
                size = random.randint(min_size,max_size)
            resize = T.Resize((size,size))
        else:
            resize = T.Resize(self.size_img)
        image = resize(image)
        mask = resize(mask)

        if self.train :
            if self.rotate:
                if random.random() > self.p_rotate:
                    if self.pi_rotate:
                        angle = int(np.random.choice([90,180,270],1,replace=True)) #Only pi/2 rotation
                        image = TF.rotate(image,angle=angle)
                        mask = TF.rotate(mask,angle=angle)
                    else:
                        if random.random() > 0.5:
                            angle = np.random.randint(0,self.angle_max)
                            image = TF.rotate(image,angle=angle,expand=True)
                            mask = TF.rotate(mask,angle=angle,expand=True)
                        else:
                            angle = np.random.randint(360-self.angle_max,360)
                            image = TF.rotate(image,angle=angle,expand=True)
                            mask = TF.rotate(mask,angle=angle,expand=True) 
            # Random crop
            i, j, h, w = T.RandomCrop.get_params(
                image, output_size=self.size_crop)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > self.p:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = to_tensor_target_lc(mask)
        
        # Apply a fixed rotation for test time:
        if self.fixing_rotate:
            image = TF.rotate(image,angle=self.angle_fix,expand=True,fill=-1,interpolation=TF.InterpolationMode.BILINEAR)
            mask = mask.unsqueeze(0)
            mask = TF.rotate(mask,angle=self.angle_fix,expand=True,fill=-1)
            mask = mask.squeeze()
        if self.normalize:
            image = TF.normalize(image,self.mean,self.std)
        
        return image, mask


    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_img,self.file_names[index]+'.jpg')).convert('RGB')
        target = Image.open(os.path.join(self.root_img,self.file_names[index]+'_m.png'))
        img, target = self.my_transform(img, target)
        return img, target

    def __len__(self):
        return len(self.file_names)
