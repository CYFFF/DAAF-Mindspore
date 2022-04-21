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
""" MGN model """
import copy

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.resnet import resnet50


class Trinet(nn.Cell):
    """ Multiple Granularity Network model

    Args:
        num_classes: number of classes
        feats: number of output features
        pool: pooling type: avg|max
        pretrained_backbone: path to pretrained resnet50 backbone
    """

    def __init__(self, num_classes=751, feats=512, pool='avg', pretrained_backbone=''):
        super().__init__()

        resnet = resnet50()
        if pretrained_backbone:
            load_param_into_net(resnet, load_checkpoint(pretrained_backbone))

        self.backone = nn.SequentialCell(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(),
            resnet.maxpool,
            resnet.layer1
        )

        self.backone2 = nn.SequentialCell(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.mean = P.ReduceMean(keep_dims=True)
        self.mask_pre = None

        self.reduction = nn.SequentialCell(
            nn.Conv2d(2048, feats, 1, has_bias=False, weight_init='HeNormal'),
            nn.BatchNorm2d(feats, gamma_init='Normal', beta_init='Zero'),
            nn.ReLU(),
        )

        self.in_planes = 2048
        self.bnneck = nn.BatchNorm1d(self.in_planes)
        self.fc_class = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')

    def construct(self, x):
        """ Forward """
        x = self.backone(x)
        x = self.backone2(x)
        out = self.mean(x, (2, 3))

        fea = self.reduction(out).squeeze(axis=3).squeeze(axis=2)
        output = self.fc_class(fea)

        return fea, output


class DAAF(nn.Cell):
    """ Multiple Granularity Network model

    Args:
        num_classes: number of classes
        feats: number of output features
        pool: pooling type: avg|max
        pretrained_backbone: path to pretrained resnet50 backbone
    """

    def __init__(self, num_classes=751, feats=512, pretrained_backbone=''):
        super().__init__()

        resnet = resnet50()
        if pretrained_backbone:
            load_param_into_net(resnet, load_checkpoint(pretrained_backbone))

        self.backone = nn.SequentialCell(
            resnet.conv1,
            resnet.bn1,
            nn.ReLU(),
            resnet.maxpool,
            resnet.layer1
        )

        self.backone2 = nn.SequentialCell(
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.mean = P.ReduceMean(keep_dims=True)
        self.mask_pre = None

        self.reduction = nn.SequentialCell(
            nn.Conv2d(2048, feats, 1, has_bias=False, weight_init='HeNormal'),
            nn.BatchNorm2d(feats, gamma_init='Normal', beta_init='Zero'),
            nn.ReLU(),
        )

        # ------------------------------  PAB ---------------------------------- #

        self.trans_conv_0 = nn.Conv2dTranspose(in_channels=340, out_channels=64, kernel_size=2, stride=2, padding=0,
                                               has_bias=True, weight_init='normal', bias_init='Zero')
        self.trans_conv_1 = nn.Conv2dTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                                               has_bias=True, weight_init='normal', bias_init='Zero')
        self.trans_conv_2 = nn.Conv2dTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                                               has_bias=True, weight_init='normal', bias_init='Zero')
        self.trans_conv_3 = nn.Conv2dTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                                               has_bias=True, weight_init='normal', bias_init='Zero')

        self.conv_0 = nn.Conv2d(in_channels=64, out_channels=5, kernel_size=1, stride=1, padding=0,
                                has_bias=True, weight_init='normal', bias_init='Zero')
        self.conv_1 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0,
                                has_bias=True, weight_init='normal', bias_init='Zero')
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0,
                                has_bias=True, weight_init='normal', bias_init='Zero')
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0,
                                has_bias=True, weight_init='normal', bias_init='Zero')
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0,
                                has_bias=True, weight_init='normal', bias_init='Zero')
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0,
                                has_bias=True, weight_init='normal', bias_init='Zero')

        self.in_planes = 2048
        self.bnneck = nn.BatchNorm1d(self.in_planes)
        self.fc_class = nn.Dense(feats, num_classes, weight_init='HeNormal', bias_init='Zero')
        self.cat = ops.Concat(axis=1)

        # ------------------------------  HAB ---------------------------------- #

        self.backbone_mask_resnet = nn.SequentialCell(
            copy.deepcopy(resnet.layer2),
            copy.deepcopy(resnet.layer3),
            copy.deepcopy(resnet.layer4)
        )

        self.trans_conv_m0 = nn.Conv2dTranspose(in_channels=2048, out_channels=64, kernel_size=2, stride=2, padding=0,
                                                has_bias=True, weight_init='normal', bias_init='Zero')

        self.trans_conv_m1 = nn.Conv2dTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                                                has_bias=True, weight_init='normal', bias_init='Zero')

        self.trans_conv_m2 = nn.Conv2dTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                                                has_bias=True, weight_init='normal', bias_init='Zero')

        self.trans_conv_m3 = nn.Conv2dTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0,
                                                has_bias=True, weight_init='normal', bias_init='Zero')

        self.conv_m0 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0,
                                 has_bias=True, weight_init='normal', bias_init='Zero')

        # --------------------------------------------------------------------- #

    def construct(self, x):
        """ Forward """
        x = self.backone(x)
        backbone_last_layer = self.backone2(x)

        keypt_group_0 = self.trans_conv_0(backbone_last_layer[:, 0:340, :, :])
        keypt_group_0 = self.trans_conv_1(keypt_group_0)
        keypt_group_0 = self.trans_conv_2(keypt_group_0)
        keypt_group_0 = self.trans_conv_3(keypt_group_0)
        keypt_group_0 = self.conv_0(keypt_group_0)

        keypt_group_1 = self.trans_conv_0(backbone_last_layer[:, 340:680, :, :])
        keypt_group_1 = self.trans_conv_1(keypt_group_1)
        keypt_group_1 = self.trans_conv_2(keypt_group_1)
        keypt_group_1 = self.trans_conv_3(keypt_group_1)
        keypt_group_1 = self.conv_1(keypt_group_1)

        keypt_group_2 = self.trans_conv_0(backbone_last_layer[:, 680:1020, :, :])
        keypt_group_2 = self.trans_conv_1(keypt_group_2)
        keypt_group_2 = self.trans_conv_2(keypt_group_2)
        keypt_group_2 = self.trans_conv_3(keypt_group_2)
        keypt_group_2 = self.conv_2(keypt_group_2)

        keypt_group_3 = self.trans_conv_0(backbone_last_layer[:, 1020:1360, :, :])
        keypt_group_3 = self.trans_conv_1(keypt_group_3)
        keypt_group_3 = self.trans_conv_2(keypt_group_3)
        keypt_group_3 = self.trans_conv_3(keypt_group_3)
        keypt_group_3 = self.conv_3(keypt_group_3)

        keypt_group_4 = self.trans_conv_0(backbone_last_layer[:, 1360:1700, :, :])
        keypt_group_4 = self.trans_conv_1(keypt_group_4)
        keypt_group_4 = self.trans_conv_2(keypt_group_4)
        keypt_group_4 = self.trans_conv_3(keypt_group_4)
        keypt_group_4 = self.conv_4(keypt_group_4)

        keypt_group_5 = self.trans_conv_0(backbone_last_layer[:, 1700:2040, :, :])
        keypt_group_5 = self.trans_conv_1(keypt_group_5)
        keypt_group_5 = self.trans_conv_2(keypt_group_5)
        keypt_group_5 = self.trans_conv_3(keypt_group_5)
        keypt_group_5 = self.conv_5(keypt_group_5)

        out = self.mean(backbone_last_layer, (2, 3))

        fea = self.reduction(out).squeeze(axis=3).squeeze(axis=2)
        output = self.fc_class(fea)

        keypt_pre = self.cat([keypt_group_0, keypt_group_1, keypt_group_2, keypt_group_3, keypt_group_4, keypt_group_5])

        VAC_resnet_output = self.backbone_mask_resnet(x)
        mask_pre = self.trans_conv_m0(VAC_resnet_output)
        mask_pre = self.trans_conv_m1(mask_pre)
        mask_pre = self.trans_conv_m2(mask_pre)
        mask_pre = self.trans_conv_m3(mask_pre)
        mask_pre = self.conv_m0(mask_pre)

        return fea, output, keypt_pre, mask_pre


class ModelLossFusion(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(ModelLossFusion, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, group_0, group_1, group_2, group_3, group_4, group_5, label):
        output = self._backbone(data)
        return self._loss_fn(output, group_0, group_1, group_2, group_3, group_4, group_5, label)
