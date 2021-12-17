import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision.models import vgg16_bn
import math
import torch

# 1、基于VGG16的全连接卷积神经网络,带bn层
class FCN_VGG16(nn.Module):
    def __init__(self, num_classes):
        super(FCN_VGG16, self).__init__()
        self.num_classes = num_classes

        # 第一个块
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn1_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn1_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 第二个块
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn2_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn2_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 第三个块
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn3_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn3_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn3_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 第四个块
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn4_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn4_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn4_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # 第五个块
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn5_1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn5_2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.bn5_3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5_1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.score1 = nn.Conv2d(512, num_classes, 1)
        self.score2 = nn.Conv2d(256, num_classes, 1)
        self.score3 = nn.Conv2d(128, num_classes, 1)

        self.conv_trans1 = nn.Conv2d(512, 256, 1)
        self.conv_trans2 = nn.Conv2d(256, num_classes, 1)

        # 转置卷积部分
        self.unsample_8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)

        self.unsample_2x_1 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)

        self.unsample_2x_2 = nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False)

    def forward(self, x):
        s1 = self.pool1_1(self.relu1_2(self.bn1_2(self.conv1_2(self.relu1_1(self.bn1_1(self.conv1_1(x)))))))
        s2 = self.pool2_1(self.relu2_2(self.bn2_2(self.conv2_2(self.relu2_1(self.bn2_1(self.conv2_1(s1)))))))
        s3 = self.pool3_1(self.relu3_3(self.bn3_3(self.conv3_3(self.relu3_2(self.bn3_2(self.conv3_2(self.relu3_1(self.bn3_1(self.conv3_1(s2))))))))))
        s4 = self.pool4_1(self.relu4_3(self.bn4_3(self.conv4_3(self.relu4_2(self.bn4_2(self.conv4_2(self.relu4_1(self.bn4_1(self.conv4_1(s3))))))))))
        s5 = self.pool5_1(self.relu5_3(self.bn5_3(self.conv5_3(self.relu5_2(self.bn5_2(self.conv5_2(self.relu5_1(self.bn5_1(self.conv5_1(s4))))))))))

        # 进行转置卷积操作
        s5 = self.unsample_2x_1(s5)
        add1 = s4 + s5

        add1 = self.conv_trans1(add1)
        add1 = self.unsample_2x_2(add1)
        add2 = add1 + s3

        output = self.conv_trans2(add2)
        output = self.unsample_8x(output)

        return output

# 2、ResNet50网络，采取跳跃连接进行反卷积
class Block(nn.Module):  # 创建块
    def __init__(self, input_channels, output_channels, stride=(1, 1), is_1x1_conv=False):
        super(Block, self).__init__()
        self.is_1x1_conv = is_1x1_conv
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-05),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False),
            nn.BatchNorm2d(input_channels, eps=1e-05),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_channels, eps=1e-05),
            nn.ReLU(inplace=True)
        )

        if is_1x1_conv:
            self.split = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(output_channels, eps=1e-05)
            )

    def forward(self, x):
        x_split = x
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        if self.is_1x1_conv:
            x_split = self.split(x)

        x_sum = x3 + x_split

        output = self.relu(x_sum)

        return output

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        )
        self.stage2 = nn.Sequential(
            Block(64, 256, is_1x1_conv=True),
            Block(256, 256, is_1x1_conv=False),
            Block(256, 256, is_1x1_conv=False)
        )
        self.stage3 = nn.Sequential(
            Block(256, 512, stride=(2, 2), is_1x1_conv=True),
            Block(512, 512, is_1x1_conv=False),
            Block(512, 512, is_1x1_conv=False),
            Block(512, 512, is_1x1_conv=False)
        )
        self.stage4 = nn.Sequential(
            Block(512, 1024, stride=(2, 2), is_1x1_conv=True),
            Block(1024, 1024, is_1x1_conv=False),
            Block(1024, 1024, is_1x1_conv=False),
            Block(1024, 1024, is_1x1_conv=False),
            Block(1024, 1024, is_1x1_conv=False),
            Block(1024, 1024, is_1x1_conv=False)
        )
        self.stage5 = nn.Sequential(
            Block(1024, 2048, stride=(2, 2), is_1x1_conv=True),
            Block(2048, 2048, is_1x1_conv=False),
            Block(2048, 2048, is_1x1_conv=False)
        )

        self.unsample2x_1 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1, bias=False)

        self.unsample2x_2 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)

        self.unsample8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)

        self.conv_trans1 = nn.Conv2d(2048, 1024, 1)

        self.conv_trans2 = nn.Conv2d(1024, 512, 1)

        self.conv_trans3 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        add0 = self.conv_trans1(s5)
        add0 = self.unsample2x_1(add0)
        add1 = add0 + s4

        add1 = self.conv_trans2(add1)
        add1 = self.unsample2x_2(add1)
        add2 = add1 + s3

        add2 = self.conv_trans3(add2)
        output = self.unsample8x(add2)

        return output

# 3、RefineNet(下采样基于ResNet50) + Attention
def un_pool(input, scale):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)  # 双线性插值进行上采样

class BasicBlock(nn.Module):  # 创建含注意力的块
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet_FCN(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_FCN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True)
        )
        self.stage2 = nn.Sequential(
            Block(64, 256, is_1x1_conv=True),
            Block(256, 256, is_1x1_conv=False),
            Block(256, 256, is_1x1_conv=False)
        )
        self.stage3 = nn.Sequential(
            Block(256, 128, stride=(2, 2), is_1x1_conv=True),
            Block(128, 512, is_1x1_conv=False),
            Block(512, 512, is_1x1_conv=False),
            Block(512, 512, is_1x1_conv=False)
        )
        self.stage4 = nn.Sequential(
            Block(512, 256, stride=(2, 2), is_1x1_conv=True),
            Block(256, 1024, is_1x1_conv=False),
            Block(256, 1024, is_1x1_conv=False),
            Block(256, 1024, is_1x1_conv=False),
            Block(256, 1024, is_1x1_conv=False),
            Block(256, 1024, is_1x1_conv=False)
        )
        self.stage5 = nn.Sequential(
            Block(1024, 512, stride=(2, 2), is_1x1_conv=True),
            Block(512, 2048, is_1x1_conv=False),
            Block(512, 2048, is_1x1_conv=False)
        )

        self.unsample2x_1 = nn.ConvTranspose2d(1024, 1024, 4, 2, 1, bias=False)

        self.unsample2x_2 = nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False)

        self.unsample8x = nn.ConvTranspose2d(num_classes, num_classes, 16, 8, 4, bias=False)

        self.conv_trans1 = nn.Conv2d(2048, 1024, 1)

        self.conv_trans2 = nn.Conv2d(1024, 512, 1)

        self.conv_trans3 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)

        add0 = self.conv_trans1(s5)
        add0 = self.unsample2x_1(add0)
        add1 = add0 + s4

        add1 = self.conv_trans2(add1)
        add1 = self.unsample2x_2(add1)
        add2 = add1 + s3

        add2 = self.conv_trans3(add2)
        output = self.unsample8x(add2)

        return output

class RefineNet_Attention(nn.Module):
    def __init__(self, block, layers, num_classes=8):
        self.inplanes = 64
        super(RefineNet_Attention, self).__init__()
        self.do = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        '''downsample block'''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        '''upsample block'''
        self.p_ims1d2_outl1_dimred = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv1_g1_pool = self._make_crp(512, 512, 3)
        self.mflow_conv1_g1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv1_g1_b3_joint_varout_dimred = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.p_ims1d2_outl2_dimred = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.adapt_stage2_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.mflow_conv1_g2_pool = self._make_crp(256, 256, 3)
        self.mflow_conv1_g2_b = self._make_rcu(256, 256, 2, 2)
        self.mflow_conv1_g2_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.p_ims1d2_outl3_dimred = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.adapt_stage3_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.mflow_conv1_g3_pool = self._make_crp(256, 256, 3)
        self.mflow_conv1_g3_b = self._make_rcu(256, 256, 2, 2)
        self.mflow_conv1_g3_b3_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.p_ims1d2_outl4_dimred = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.adapt_stage4_b = self._make_rcu(256, 256, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.mflow_conv1_g4_pool = self._make_crp(256, 256, 3)
        self.mflow_conv1_g4_b = self._make_rcu(256, 256, 2, 2)

        self.clf_conv = nn.Conv2d(256, num_classes, kernel_size=3, stride=1, padding=1, bias=True)

    def _make_crp(self, inplanes, planes, stages):
        layers = [CRPBlock(inplanes, planes, stages)]
        return nn.Sequential(*layers)

    def _make_rcu(self, inplanes, planes, blocks, stages):
        layers = [RCUBlock(inplanes, planes, blocks, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(blocks):
            layers.append((block(self.inplanes, planes)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        '''downsample process'''
        l1 = self.layer1(x1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        '''upsample process'''
        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv1_g1_pool(x4)
        x4 = self.mflow_conv1_g1_b(x4)
        x4 = self.mflow_conv1_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv1_g2_pool(x3)
        x3 = self.mflow_conv1_g2_b(x3)
        x3 = self.mflow_conv1_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode='bilinear', align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x3 + x2
        x2 = F.relu(x2)
        x2 = self.mflow_conv1_g3_pool(x2)
        x2 = self.mflow_conv1_g3_b(x2)
        x2 = self.mflow_conv1_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode='bilinear', align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv1_g4_pool(x1)
        x1 = self.mflow_conv1_g4_b(x1)
        x1 = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)(x1)
        # x1 = self.do(x1)

        out = self.clf_conv(x1)

        return out

stages_suffixes = {0: '_conv', 1: '_conv_relu_varout_dimred'}

class RCUBlock(nn.Module):
    def __init__(self, inplanes, planes, n_blocks, n_stages):
        super(RCUBlock, self).__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}{}'.format(i+1, stages_suffixes[j]),
                        nn.Conv2d(inplanes if (i == 0) and (j == 0) else planes,
                        planes, kernel_size=3, stride=1, padding=1,
                        bias=(j == 0)))

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = F.relu(x)
                x = getattr(self, '{}{}'.format(i+1, stages_suffixes[j]))(x)
            x += residual
        return x

class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feature, *shapes):
        super(MultiResolutionFusion, self).__init__()

        _, max_h, max_w = max(shapes, key=lambda x: x[1])

        self.scale_factors = []

        for i, shape in enumerate(shapes):
            feat, h, w = shape
            if max_h % h != 0:
                raise ValueError('max_size not divisble by shape {}'.format(i))

            self.scale_factors.append(max_h // h)
            self.add_module(
                'resolve{}'.format(i),
                nn.Conv2d(feat, out_feature, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )

    def forward(self, *xs):
        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = un_pool(output, self.scale_factors[0])

        for i, x in enumerate(xs[1:], 1):
            tmp_out = self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                tmp_out = un_pool(tmp_out, self.scale_factors[i])
            output = output + tmp_out

        return output

class CRPBlock(nn.Module):
    def __init__(self, inplanes, planes, n_stages):
        super(CRPBlock, self).__init__()

        for i in range(n_stages):
            setattr(self, '{}_{}'.format(i+1, 'outvar_dimred'),
                    nn.Conv2d(inplanes if (i == 0) else planes,
                              planes, kernel_size=3, stride=1, padding=1,
                              bias=False))

        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):

        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i+1, 'outvar_dimred'))(top)
            x = top + x

        return x

def rf50(num_classes, imagenet=False, **kwargs):
    model = RefineNet_Attention(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    return model

class ChannelAttention(nn.Module): # 通道注意力机制
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module): # 空间注意力机制
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

