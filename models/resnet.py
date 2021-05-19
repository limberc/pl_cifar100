import os
import random
import sys

import torch
import torch.nn as nn
import numpy as np
from torchvision.models.resnet import ResNet,BasicBlock

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_lambda(alpha=1.0):
    '''Return lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    return lam


class CIFARResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__(block, layers, num_classes, zero_init_residual, groups, width_per_group,
                         replace_stride_with_dilation,
                         norm_layer)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ResNetBasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResNetBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * ResNetBasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != ResNetBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBasicBlock.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * ResNetBasicBlock.expansion)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNetBottleneck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * ResNetBottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * ResNetBottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * ResNetBottleneck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBottleneck.expansion, stride=stride, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * ResNetBottleneck.expansion)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dropout=False, num_classes=100, zero_init_residual=False,
                 norm_layer=None, random_layers=[0, 1, 2, 3], method='mixup', mixup_alpha=0., beta=1.0):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = 64

        if num_classes < 200:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True))
        elif num_classes == 200:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2)
            )
        elif num_classes == 1000:
            self.layer0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                          bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        # we use a different inputsize than the original paper
        # so layer1's stride is 1
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Mixup Staff
        self.beta = beta
        self.method = method
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.random_layers = random_layers
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResNetBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, ResNetBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def prepare(self, x):
        def fhook(module, input, output):
            # can not modify the input
            if self.training:
                bs = output.shape[0]
                group1 = output[:bs // 2]
                group2 = output[bs // 2:]
                try:
                    fhandle.remove()
                except AttributeError:
                    pass
                return module.lam * group1 + (1 - module.lam) * group2
            return output

        def bphook(module, grad_input, grad_output):
            # can not modify the input
            if self.training:
                assert len(grad_input) == 2 and grad_input[0].shape == grad_input[1].shape
                fake_grad_input = module.lam * grad_input[0] + (1 - module.lam) * grad_input[1]
                # fake_grad_input /= 2. Not helped.
                # Smart Remove.
                try:
                    bhandle.remove()
                except AttributeError:
                    pass
                return (fake_grad_input, fake_grad_input)
            return grad_input

        if ('mixup' in self.method) and self.training:
            if self.method == 'mixup':
                self.layer_mix = 0
            elif self.method in ['mixup_hidden', 'grad_mixup_random']:
                assert set(self.random_layers).issubset(
                    [0, 1, 2, 3]), "The range should be subset of [0, 1, 2, 3], not support {}".format(
                    self.random_layers)
                self.layer_mix = random.choice(self.random_layers)
            for name, layer in self.named_modules():
                if name == "layer{}".format(self.layer_mix):
                    self.lam = layer.lam = get_lambda(self.mixup_alpha)
            for name, layer in self.named_modules():
                if name == "layer{}".format(self.layer_mix) and self.layer_mix != 0:
                    if self.method in ['mixup', 'mixup_hidden', 'grad_mixup_random']:
                        fhandle = layer.register_forward_hook(fhook)
                    if self.method == 'grad_mixup_random':
                        bhandle = layer.register_backward_hook(bphook)
            if self.layer_mix == 0:
                # Input Mixup
                x = x[:x.size(0) // 2] * self.lam + x[x.size(0) // 2:] * (1 - self.lam)
        return x

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prepare(x)  # Update all params on the fly.
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes, random_layers=[0, 1, 2, 3], method='mixup', mixup_alpha=1.0, beta=1.0,
             dropout=False):
    model = ResNet(ResNetBasicBlock, [2, 2, 2, 2], num_classes=num_classes, method=method,
                   random_layers=random_layers, mixup_alpha=mixup_alpha, beta=beta, dropout=dropout)
    return model


def resnet34(num_classes, random_layers=[0, 1, 2, 3], method='mixup', mixup_alpha=1.0, beta=1.0,
             dropout=False):
    # model = ResNet(ResNetBasicBlock, [3, 4, 6, 3], num_classes=num_classes, method=method,
    #                random_layers=random_layers, mixup_alpha=mixup_alpha, beta=beta,
    #                dropout=dropout)
    # return model
    return CIFARResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes, random_layers=[0, 1, 2, 3], method='mixup', mixup_alpha=1.0, beta=1.0,
             dropout=False):
    model = ResNet(ResNetBottleneck, [3, 4, 6, 3], num_classes=num_classes, method=method,
                   random_layers=random_layers, mixup_alpha=mixup_alpha, beta=beta,
                   dropout=dropout)
    return model


def resnet101(num_classes, random_layers=[0, 1, 2, 3], method='mixup', mixup_alpha=1.0, beta=1.0,
              dropout=False):
    model = ResNet(ResNetBottleneck, [3, 4, 23, 3], num_classes=num_classes, method=method,
                   random_layers=random_layers, mixup_alpha=mixup_alpha, beta=beta,
                   dropout=dropout)
    return model
