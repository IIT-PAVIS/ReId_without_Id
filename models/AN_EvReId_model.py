import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from models.ResNet import *


class ANmodel(torch.nn.Module):
    def __init__(self):
        super(ANmodel, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(5, 64, 3, stride=1, padding=1),  #
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 5, 3, stride=1, padding=2),  # b, 8, 3, 3
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return decoded


def weights_init_kaiming(m):

    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block_1 = []
        add_block_1 += [nn.Linear(input_dim, num_bottleneck)]
        add_block_1 += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block_1 += [nn.LeakyReLU(0.5)]
            #add_block_1 += [nn.SELU()]
        if dropout:
            add_block_1 += [nn.Dropout(p=0.3)]
        add_block_1 = nn.Sequential(*add_block_1)
        add_block_1.apply(weights_init_kaiming)

        classifier_1 = []
        classifier_1 += [nn.Linear(num_bottleneck, class_num)]
        classifier_1 = nn.Sequential(*classifier_1)
        classifier_1.apply(weights_init_classifier)

        self.add_block_1 = add_block_1
        self.classifier_1 = classifier_1

    def forward(self, x):
        x = self.add_block_1(x)
        x = self.classifier_1(x)

        return x


class EvReId(nn.Module):
    def __init__(self, class_num=22, num_channel=None, AE_block=None):
        super().__init__()

        #if layer == '50':
        backbone = resnet50(pretrained=True, num_ch=num_channel)
        #elif layer == '152':
        #backbone = ResNet.resnet152(pretrained=True)
        #self.ae_block = True
        self.model_ae = ANmodel()
        # avg pooling to global pooling
        backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = backbone
        #self.block_num = block_num
        self.classifier = ClassBlock(2048, class_num, dropout=False, relu=True, num_bottleneck=256)

    def forward(self, x, path=None):

        """Auto Encoder"""
        #voxel_reconst = None
        #if self.ae_block:
        x = self.model_ae(x)
        voxel_reconst = x.clone()

        """ReId"""
        if x.dim() == 5:
            x = x.unsqueeze(0)
        #pdb.set_trace()
        x = self.model.conv1(x)
        #pdb.set_trace()
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)

        x = self.model.layer2(x)

        x = self.model.layer3(x)

        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        feature = self.classifier.add_block_1(x)
        category = self.classifier.classifier_1(feature)

        return category, feature, x, voxel_reconst
