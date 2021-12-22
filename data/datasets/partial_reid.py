# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com

add by chenyifan
chenyifan0627@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class PartialREID(BaseImageDataset):
    """
    Partial_REID
    Reference:
    @inproceedings{zheng2015partial,
        title={Partial person re-identification},
        author={Zheng, Wei-Shi and Li, Xiang and Xiang, Tao and Liao, Shengcai and Lai, Jianhuang and Gong, Shaogang},
        booktitle={Proceedings of the IEEE International Conference on Computer Vision},
        pages={4678--4686},
        year={2015}
    }


    Dataset statistics:
    # identities: 60
    # images: (train on Market-1501) + 60*5 (query) + 60*5 (gallery)
    """
    Partial_REID_dataset_dir = 'PartialREID'
    Market_1501_dataset_dir = 'Market-1501'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(PartialREID, self).__init__()
        self.Partial_REID_dataset_dir = osp.join(root, self.Partial_REID_dataset_dir)
        self.Market_1501_dataset_dir = osp.join(root, self.Market_1501_dataset_dir)
        self.train_dir = osp.join(self.Market_1501_dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.Partial_REID_dataset_dir, 'partial_body_images')
        self.gallery_dir = osp.join(self.Partial_REID_dataset_dir, 'whole_body_images')

        self._check_before_run()

        train = self._process_market_1501_dir(self.train_dir, relabel=True)
        query = self._process_partial_REID_query_dir(self.query_dir, relabel=False)
        gallery = self._process_partial_REID_gallery_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.Partial_REID_dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.Partial_REID_dataset_dir))
        if not osp.exists(self.Market_1501_dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.Market_1501_dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_market_1501_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_partial_REID_query_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid = map(int, pattern.search(img_path).groups())
            # if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = int(pattern.search(img_path).groups()[0])
            camid = 1
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 119  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _process_partial_REID_gallery_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid = map(int, pattern.search(img_path).groups())
            # if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid = int(pattern.search(img_path).groups()[0])
            camid = 2
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 119  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset