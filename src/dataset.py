# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" ReID dataset processing """

import math
import os
import random
import re

import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
import numpy as np
from PIL import Image

from src.sampler import ReIDDistributedSampler


def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    """ Filter images be extension

    Args:
        directory: path to images folder
        ext: extensions split by "|"

    Returns:
        list of image filenames
    """
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class Market1501:
    """
    Market1501 dataset

    Args:
        datadir: path to dataset
        data_part: part of data to usage (one of train|test|query)
        transform:
    """
    def __init__(self, datadir, data_part='train', transform=None):
        self.transform = transform

        data_path = datadir
        if data_part == 'train':
            data_path += '/bounding_box_train'
        elif data_part == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}


    def __getitem__(self, index):
        """ Get image and label by index """
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        img = np.asarray(img)

        return img, target


    def __len__(self):
        """ Dataset length """
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """ Get person id by path
        Args:
            file_path: unix style file path

        Returns:
            person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """ Get camera id by path
        Args:
            file_path: unix style file path

        Returns:
            camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """ Person id list corresponding to dataset image paths  """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """ Unique person ids in ascending order """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """ Camera id list corresponding to dataset image paths """
        return [self.camera(path) for path in self.imgs]


class Market1501_kpt_mask:
    """
    Market1501 dataset

    Args:
        datadir: path to dataset
        data_part: part of data to usage (one of train|test|query)
        transform:
    """
    def __init__(self, datadir, data_part='train', transform=None):
        self.transform = transform

        data_path = datadir
        if data_part == 'train':
            data_path += '/bounding_box_train'
        elif data_part == 'test':
            data_path += '/bounding_box_test'
        else:
            data_path += '/query'

        self.imgs = [path for path in list_pictures(data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}


    def __getitem__(self, index):
        """ Get image and label by index """
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        img = np.asarray(img)


        mask_path = path.replace('Market-1501', 'mask-anno')
        mask = Image.open(mask_path).convert('L').resize((128, 256))
        threshold = 100
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)
        mask_bin = mask.point(table, '1')
        mask_np = np.asarray(mask_bin).astype(np.float32)

        keypt_path = path.replace('Market-1501', 'Market_cpn_keypoints')
        keypt_path = keypt_path.replace('bounding_box_train', 'bounding_box_train_256_2')
        keypt_path = keypt_path.replace('.jpg', '')
        keypt_mask_all = []
        for i in range(17):
            keypt_path_temp = keypt_path + '_' + '%02d' % (i) + '.png'
            keypt = Image.open(keypt_path_temp).convert('L')
            # keypt = self.Resize(keypt)
            # if if_flip:
            #     keypt = F.hflip(keypt)
            # keypt = self.Pad(keypt)
            # keypt = F.crop(keypt, random_crop_i, random_crop_j, random_crop_h, random_crop_w)
            # keypt = self.To_tensor(keypt)
            # keypt = torch.unsqueeze(keypt, 0)
            keypt_mask_all.append(np.asarray(keypt).astype(np.float32))

        keypt_mask_all.append(mask_np)
        keypt_mask = np.stack(keypt_mask_all, axis=2)


        group_0 = keypt_mask[:, :, 0:3]
        group_1 = keypt_mask[:, :, 3:6]
        group_2 = keypt_mask[:, :, 6:9]
        group_3 = keypt_mask[:, :, 9:12]
        group_4 = keypt_mask[:, :, 12:15]
        group_5 = keypt_mask[:, :, 15:18]

        return img, group_0, group_1, group_2, group_3, group_4, group_5, target


    def __len__(self):
        """ Dataset length """
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """ Get person id by path
        Args:
            file_path: unix style file path

        Returns:
            person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """ Get camera id by path
        Args:
            file_path: unix style file path

        Returns:
            camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """ Person id list corresponding to dataset image paths  """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """ Unique person ids in ascending order """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """ Camera id list corresponding to dataset image paths """
        return [self.camera(path) for path in self.imgs]


def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
    """ Random erasing augmentation

    Args:
        img: input image
        probability: augmentation probability
        sl: min erasing area
        sh: max erasing area
        r1: erasing ratio
        mean: erasing color
    Returns:
        augmented image
    """
    if random.uniform(0, 1) > probability:
        return img

    ch, height, width = img.shape

    for _ in range(100):
        area = height * width

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < width and h < height:
            x1 = random.randint(0, height - h)
            y1 = random.randint(0, width - w)
            if ch == 3:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
                img[1, x1:x1 + h, y1:y1 + w] = mean[1]
                img[2, x1:x1 + h, y1:y1 + w] = mean[2]
            else:
                img[0, x1:x1 + h, y1:y1 + w] = mean[0]
            return img

    return img


def create_dataset(
        image_folder,
        ims_per_id=4,
        ids_per_batch=32,
        batch_size=None,
        rank=0,
        group_size=1,
        resize_h_w=(384, 128),
        num_parallel_workers=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        data_part='train',
):
    """ Crate dataloader for ReID

    Args:
        image_folder: path to image folder
        ims_per_id: number of ids in batch
        ids_per_batch: number of imager per id
        batch_size: batch size (if None then batch_size=ims_per_id*ids_per_batch)
        rank: process id
        group_size: device number
        resize_h_w: height and width of image
        num_parallel_workers: number of parallel workers
        mean: image mean value for normalization
        std: image std value for normalization
        data_part: part of data: train|test|query

    Returns:
        if train data_part:
            dataset
        else:
            dataset, camera_ids, person_ids
    """
    mean = [m * 255 for m in mean]
    std = [s * 255 for s in std]

    if batch_size is None:
        batch_size = ids_per_batch * ims_per_id

    reid_dataset = Market1501(image_folder, data_part=data_part)

    sampler, shuffle = None, None

    if data_part == 'train':
        sampler = ReIDDistributedSampler(
            reid_dataset,
            batch_id=ids_per_batch,
            batch_image=ims_per_id,
            rank=rank,
            group_size=group_size,
        )

        transforms_list = [
            C.Resize(resize_h_w),
            C.RandomHorizontalFlip(),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]
    else:
        shuffle = False

        transforms_list = [
            C.Resize(resize_h_w),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]

    dataset = ds.GeneratorDataset(
        source=reid_dataset,
        column_names=['image', 'label'],
        sampler=sampler,
        shuffle=shuffle,
    )

    dataset = dataset.map(
        operations=transforms_list,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
    )

    if data_part == 'train':
        dataset = dataset.map(
            operations=random_erasing,
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if data_part == 'train':
        return dataset
    return dataset, reid_dataset.cameras, reid_dataset.ids


def create_dataset_DAAF_training(
        image_folder,
        ims_per_id=4,
        ids_per_batch=32,
        batch_size=None,
        rank=0,
        group_size=1,
        resize_h_w=(256, 128),
        num_parallel_workers=32,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        data_part='train',
):
    """ Crate dataloader for ReID

    Args:
        image_folder: path to image folder
        ims_per_id: number of ids in batch
        ids_per_batch: number of imager per id
        batch_size: batch size (if None then batch_size=ims_per_id*ids_per_batch)
        rank: process id
        group_size: device number
        resize_h_w: height and width of image
        num_parallel_workers: number of parallel workers
        mean: image mean value for normalization
        std: image std value for normalization
        data_part: part of data: train|test|query

    Returns:
        if train data_part:
            dataset
        else:
            dataset, camera_ids, person_ids
    """
    if data_part != 'train':
        raise Exception('only support training set')

    mean = [m * 255 for m in mean]
    std = [s * 255 for s in std]

    if batch_size is None:
        batch_size = ids_per_batch * ims_per_id

    reid_dataset = Market1501_kpt_mask(image_folder, data_part=data_part)
    reid_dataset.__getitem__(index=1)
    sampler, shuffle = None, None

    if data_part == 'train':
        sampler = ReIDDistributedSampler(
            reid_dataset,
            batch_id=ids_per_batch,
            batch_image=ims_per_id,
            rank=rank,
            group_size=group_size,
        )

        transforms_list = [
            C.Resize(resize_h_w),
        ]
        transforms_list_0 = [
            C.Resize((int(resize_h_w[0]/2),int(resize_h_w[1]/2))),
        ]
        transforms_list_1 = [
            C.RandomHorizontalFlip(),
        ]
        transforms_list_2 = [
            C.Normalize(mean=mean, std=std),
        ]
        transforms_list_3 = [
            C.HWC2CHW(),
        ]
        # transforms_list_2 = [C.HWC2CHW(),]
    else:
        shuffle = False

        transforms_list = [
            C.Resize(resize_h_w),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW(),
        ]

    dataset = ds.GeneratorDataset(
        source=reid_dataset,
        column_names=['image', 'group_0', 'group_1', 'group_2', 'group_3', 'group_4', 'group_5', 'label'],
        sampler=sampler,
        shuffle=shuffle,
    )

    dataset = dataset.map(
        operations=transforms_list,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_0,
        input_columns=["group_0"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_0,
        input_columns=["group_1"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_0,
        input_columns=["group_2"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_0,
        input_columns=["group_3"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_0,
        input_columns=["group_4"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_0,
        input_columns=["group_5"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_1,
        input_columns=["image", 'group_0', 'group_1', 'group_2', 'group_3', 'group_4', 'group_5'],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_2,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["image"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["group_0"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["group_1"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["group_2"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["group_3"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["group_4"],
        num_parallel_workers=num_parallel_workers,
    )

    dataset = dataset.map(
        operations=transforms_list_3,
        input_columns=["group_5"],
        num_parallel_workers=num_parallel_workers,
    )

    # # , "keypt"
    # dataset = dataset.map(
    #     operations=transforms_list_2,
    #     input_columns=["image", 'group_0', 'group_1', 'group_2', 'group_3', 'group_4', 'group_5',],
    #     num_parallel_workers=num_parallel_workers,
    # )

    if data_part == 'train':
        dataset = dataset.map(
            operations=random_erasing,
            input_columns=["image"],
            num_parallel_workers=num_parallel_workers,
        )

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if data_part == 'train':
        return dataset
    return dataset, reid_dataset.cameras, reid_dataset.ids