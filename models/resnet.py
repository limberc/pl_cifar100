import torch
from torch import nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

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


def resnet18(num_classes=100):
    """ return a ResNet 18 object
    """
    return CIFARResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=100):
    """ return a ResNet 34 object
    """
    return CIFARResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=100):
    """ return a ResNet 50 object
    """
    return CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=100):
    """ return a ResNet 101 object
    """
    return CIFARResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=100):
    """ return a ResNet 152 object
    """
    return CIFARResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def resnext50(num_classes=100):
    return CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes, groups=32, width_per_group=4)


def resnext101(num_classes=100):
    return CIFARResNet(Bottleneck, [3, 4, 23, 3], num_classes, groups=32, width_per_group=8)


def wideresnet50(num_classes=100):
    return CIFARResNet(Bottleneck, [3, 4, 6, 3], num_classes, width_per_group=64 * 2)


def wideresnet101(num_classes=100):
    return CIFARResNet(Bottleneck, [3, 4, 23, 3], num_classes, width_per_group=64 * 2)
