# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck

import copy

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)


        # --------------------------------  part-aligned constraint ------------------------------------ #

        self.trans_conv_0 = torch.nn.ConvTranspose2d(in_channels=340, out_channels=64, kernel_size=2, stride=2,
                                                     padding=0)
        nn.init.normal_(self.trans_conv_0.weight, std=0.001)
        nn.init.constant_(self.trans_conv_0.bias, 0)

        self.trans_conv_1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2,
                                                     padding=0)
        nn.init.normal_(self.trans_conv_1.weight, std=0.001)
        nn.init.constant_(self.trans_conv_1.bias, 0)

        self.trans_conv_2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2,
                                                     padding=0)
        nn.init.normal_(self.trans_conv_2.weight, std=0.001)
        nn.init.constant_(self.trans_conv_2.bias, 0)

        self.trans_conv_3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2,
                                                     padding=0)
        nn.init.normal_(self.trans_conv_3.weight, std=0.001)
        nn.init.constant_(self.trans_conv_3.bias, 0)

        self.conv_0 = torch.nn.Conv2d(in_channels=64, out_channels=5, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_0.weight, std=0.001)
        nn.init.constant_(self.conv_0.bias, 0)

        self.conv_1 = torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_1.weight, std=0.001)
        nn.init.constant_(self.conv_1.bias, 0)

        self.conv_2 = torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_2.weight, std=0.001)
        nn.init.constant_(self.conv_2.bias, 0)

        self.conv_3 = torch.nn.Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_3.weight, std=0.001)
        nn.init.constant_(self.conv_3.bias, 0)

        self.conv_4 = torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_4.weight, std=0.001)
        nn.init.constant_(self.conv_4.bias, 0)

        self.conv_5 = torch.nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_5.weight, std=0.001)
        nn.init.constant_(self.conv_5.bias, 0)

        # ---------------------------------------------------------------------------------------------- #

        # ------------------------------  visual attention constraint ---------------------------------- #

        self.backbone_mask_resnet = nn.Sequential(
            copy.deepcopy(self.base.layer2),
            copy.deepcopy(self.base.layer3),
            copy.deepcopy(self.base.layer4)
        )

        self.trans_conv_m0 = torch.nn.ConvTranspose2d(in_channels=2048, out_channels=64, kernel_size=2, stride=2,
                                                      padding=0)
        nn.init.normal_(self.trans_conv_0.weight, std=0.001)
        nn.init.constant_(self.trans_conv_0.bias, 0)

        self.trans_conv_m1 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2,
                                                      padding=0)
        nn.init.normal_(self.trans_conv_1.weight, std=0.001)
        nn.init.constant_(self.trans_conv_1.bias, 0)

        self.trans_conv_m2 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2,
                                                      padding=0)
        nn.init.normal_(self.trans_conv_2.weight, std=0.001)
        nn.init.constant_(self.trans_conv_2.bias, 0)

        self.trans_conv_m3 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2,
                                                      padding=0)
        nn.init.normal_(self.trans_conv_3.weight, std=0.001)
        nn.init.constant_(self.trans_conv_3.bias, 0)

        self.conv_m0 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        nn.init.normal_(self.conv_0.weight, std=0.001)
        nn.init.constant_(self.conv_0.bias, 0)

        # ---------------------------------------------------------------------------------------------- #

    def forward(self, x):
        backbone_last_layer, resnet_layer_1_outputs = self.base(x)

        # --------------------------------  part-aligned constraint ------------------------------------ #

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

        keypt_pre = torch.cat([keypt_group_0, keypt_group_1, keypt_group_2, keypt_group_3, keypt_group_4, keypt_group_5], dim=1)

        # --------------------------------------------------------------------------------------------- #

        # ------------------------------ visual attention constraint ---------------------------------- #

        VAC_resnet_output = self.backbone_mask_resnet(resnet_layer_1_outputs)
        mask_pre = self.trans_conv_m0(VAC_resnet_output)
        mask_pre = self.trans_conv_m1(mask_pre)
        mask_pre = self.trans_conv_m2(mask_pre)
        mask_pre = self.trans_conv_m3(mask_pre)
        mask_pre = self.conv_m0(mask_pre)

        # --------------------------------------------------------------------------------------------- #


        global_feat = self.gap(backbone_last_layer)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, keypt_pre, mask_pre  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
