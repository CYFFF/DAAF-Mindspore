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
""" Evaluation script """

import os
import time

import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore import numpy as mnp
from mindspore.common import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from scipy.spatial.distance import cdist

from metric_utils.functions import cmc, mean_ap
from metric_utils.re_ranking import re_ranking
from model_utils.config import get_config
from model_utils.device_adapter import get_device_id, get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.dataset import create_dataset
from src.network import MGN, DAAF

set_seed(1)
config = get_config()


def modelarts_pre_process():
    """ Modelarts pre process function """
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if config.device_target == "GPU":
            init()
            device_id = get_rank()
            device_num = get_group_size()
        elif config.device_target == "Ascend":
            device_id = get_device_id()
            device_num = get_device_num()
        else:
            raise ValueError("Not support device_target.")

        # Each server contains 8 devices as most.
        if device_id % min(device_num, 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(device_id, zip_file_1, save_dir_1))

    config.log_path = os.path.join(config.output_path, config.log_path)


def extract_feature(model, dataset):
    """ Extract dataset features from model """
    def fliphor(tensor):
        """ Flip tensor """
        return tensor[..., ::-1].copy()

    data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

    features = []

    for data in data_loader:
        images_ = data["image"]

        ff = mnp.zeros((images_.shape[0], 512))
        for i in range(2):
            if i == 1:
                images_ = fliphor(images_)
            images = Tensor.from_numpy(images_)
            outputs = model(images)
            f = outputs[0]
            ff = ff + f

        fnorm = mnp.sqrt((ff ** 2).sum(axis=1, keepdims=True))
        ff = ff / fnorm.expand_as(ff)

        features.append(ff.asnumpy())

    return np.concatenate(features, axis=0)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    """ Run evaluation """
    # re_rank = True
    re_rank = False

    config.image_size = list(map(int, config.image_size.split(',')))
    config.image_mean = list(map(float, config.image_mean.split(',')))
    config.image_std = list(map(float, config.image_std.split(',')))

    _enable_graph_kernel = False
    context.set_context(
        mode=context.GRAPH_MODE,
        enable_graph_kernel=_enable_graph_kernel,
        device_target=config.device_target,
    )

    config.rank = 0
    config.device_id = get_device_id()
    config.group_size = 1

    t_dataset, t_cams, t_ids = create_dataset(
        config.data_dir,
        ims_per_id=config.ims_per_id,
        ids_per_batch=config.ids_per_batch,
        mean=config.image_mean,
        std=config.image_std,
        resize_h_w=config.image_size,
        batch_size=config.per_batch_size,
        rank=config.rank,
        group_size=config.group_size,
        data_part='test'
    )

    q_dataset, q_cams, q_ids = create_dataset(
        config.data_dir,
        ims_per_id=config.ims_per_id,
        ids_per_batch=config.ids_per_batch,
        mean=config.image_mean,
        std=config.image_std,
        resize_h_w=config.image_size,
        batch_size=config.per_batch_size,
        rank=config.rank,
        group_size=config.group_size,
        data_part='query'
    )

    # network = MGN(num_classes=config.n_classes)
    network = DAAF(num_classes=config.n_classes)

    # pre_trained
    if config.eval_model:
        print('Load model from', config.eval_model)
        ret = load_param_into_net(network, load_checkpoint(config.eval_model))
        print(ret)
    else:
        print('PRETRAINED MODEL NOT SELECTED!!!')

    gf = extract_feature(network, t_dataset)
    print('Got gallery features')
    qf = extract_feature(network, q_dataset)
    print('Got query features')

    if re_rank:
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
    else:
        dist = cdist(qf, gf)
    r = cmc(dist, q_ids, t_ids, q_cams, t_cams,
            separate_camera_set=False,
            single_gallery_shot=False,
            first_match_break=True)
    m_ap = mean_ap(dist, q_ids, t_ids, q_cams, t_cams)

    print(
        '[INFO] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'.format(
            m_ap,
            r[0], r[2], r[4], r[9],
        )
    )


if __name__ == '__main__':
    run_eval()
