# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
# from torch.utils.data import Dataset
import re
import data.torchvision_transforms_modify.transforms as T
import data.torchvision_transforms_modify.functional as F
# import torch
from data.transforms.transforms import RandomErasing


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            # image path = /home/chenyf/Market-1501/bounding_box_train/0984_c1s4_064686_03.jpg
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset():
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class DatasetGenerator:
    def __init__(self):
        self.data = np.random.sample((5, 2))
        self.label = np.random.sample((5, 1))

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)



class ImageDatasetTrain():
    """Image Person ReID Dataset with Keypoints"""

    def __init__(self, cfg, dataset, transform=None):
        self.cfg = cfg
        self.dataset = dataset
        self.transform = transform
        # self.PIL_to_tensor = T.Compose([T.ToTensor()])
        self.To_tensor = T.ToTensor()
        self.Resize = T.Resize(cfg.INPUT.SIZE_TRAIN)
        self.RandomHorizontalFlipReturnValue = T.RandomHorizontalFlipReturnValue(p=cfg.INPUT.PROB)
        self.Pad = T.Pad(cfg.INPUT.PADDING)
        self.RandomCropReturnValue = T.RandomCropReturnValue(cfg.INPUT.SIZE_TRAIN)
        self.Normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        self.RandomErasing = RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        # Transform
        img = self.Resize(img)
        img, if_flip = self.RandomHorizontalFlipReturnValue(img)
        img = self.Pad(img)
        img, random_crop_i, random_crop_j, random_crop_h, random_crop_w = self.RandomCropReturnValue(img)
        img = self.To_tensor(img)
        img = self.Normalize(img)
        img = self.RandomErasing(img)

        keypt_path = img_path.replace('Market-1501', 'Market_cpn_keypoints')
        keypt_path = keypt_path.replace('bounding_box_train', 'bounding_box_train_256_2')
        # keypt_path = keypt_path.replace('bounding_box_train', 'bounding_box_train_256')
        keypt_path = keypt_path.replace('.jpg', '')
        for i in range(17):
            keypt_path_temp = keypt_path + '_' + '%02d' % (i) + '.png'
            keypt = Image.open(keypt_path_temp).convert('L')
            keypt = self.Resize(keypt)
            if if_flip:
                keypt = F.hflip(keypt)
            keypt = self.Pad(keypt)
            keypt = F.crop(keypt, random_crop_i, random_crop_j, random_crop_h, random_crop_w)
            keypt = self.To_tensor(keypt)
            # keypt = torch.unsqueeze(keypt, 0)
            if i == 0:
                keypt_all = keypt
            else:
                keypt_all = torch.cat((keypt_all, keypt), 0)

        mask_path = img_path.replace('Market-1501', 'mask-anno')
        mask = Image.open(mask_path).convert('L')
        threshold = 100
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)
        mask_bin = mask.point(table, '1')

        mask_bin = self.Resize(mask_bin)
        if if_flip:
            mask_bin = F.hflip(mask_bin)
        mask_bin = self.Pad(mask_bin)
        mask_bin = F.crop(mask_bin, random_crop_i, random_crop_j, random_crop_h, random_crop_w)
        mask_bin = self.To_tensor(mask_bin)


            # keypt_all = self.PIL_to_tensor(keypt_PIL)
            # all_tensor = torch.cat((all_tensor, keypt_tensor), 0)

        # if self.transform is not None:
        #     img = self.transform(img)

        # keypts = Image.open(img_path)
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, pid, camid, img_path, keypt_all, mask_bin
