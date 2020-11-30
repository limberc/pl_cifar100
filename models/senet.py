from timm.models.registry import register_model
from timm.models.senet import SENet, build_model_with_cfg, SEResNetBlock, SEBottleneck, \
    SEResNeXtBottleneck, SEResNetBottleneck
from torch import nn


class CIFARSENet(SENet):
    def __init__(self, block, layers, groups, reduction, drop_rate=0.2,
                 in_chans=3, inplanes=64, input_3x3=False, downsample_kernel_size=1,
                 downsample_padding=0, num_classes=1000, global_pool='avg'):
        super().__init__(block, layers, groups, reduction, drop_rate, in_chans, inplanes, input_3x3,
                         downsample_kernel_size, downsample_padding, num_classes, global_pool)
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward_features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def cifar_cfg(num_classes, **kwargs):
    if num_classes == 10:
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif num_classes == 100:
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]

    return {
        'num_classes': num_classes, 'input_size': (3, 40, 40), 'pool_size': (3, 3),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': mean, 'std': std,
        'first_conv': 'layer0.conv1', 'classifier': 'last_linear',
        **kwargs
    }


def _create_senet(num_classes, variant, **kwargs):
    return build_model_with_cfg(
        CIFARSENet, variant, default_cfg=cifar_cfg(num_classes=num_classes),
        pretrained=False, **kwargs)


@register_model
def cifar_seresnet18(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNetBlock, layers=[2, 2, 2, 2], groups=1, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnet18', **model_args)


@register_model
def cifar_seresnet34(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNetBlock, layers=[3, 4, 6, 3], groups=1, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnet34', **model_args)


@register_model
def cifar_seresnet50(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 4, 6, 3], groups=1, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnet50', **model_args)


@register_model
def cifar_seresnet101(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 4, 23, 3], groups=1, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnet101', **model_args)


@register_model
def cifar_seresnet152(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNetBottleneck, layers=[3, 8, 36, 3], groups=1, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnet152', **model_args)


@register_model
def cifar_senet154(num_classes=100, **kwargs):
    model_args = dict(
        block=SEBottleneck, layers=[3, 8, 36, 3], groups=64, reduction=16,
        downsample_kernel_size=3, downsample_padding=1, inplanes=128, input_3x3=True, **kwargs)
    return _create_senet(num_classes, 'cifar_senet154', **model_args)


@register_model
def cifar_seresnext26_32x4d(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNeXtBottleneck, layers=[2, 2, 2, 2], groups=32, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnext26_32x4d', **model_args)


@register_model
def cifar_seresnext50_32x4d(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNeXtBottleneck, layers=[3, 4, 6, 3], groups=32, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnext50_32x4d', **model_args)


@register_model
def cifar_seresnext101_32x4d(num_classes=100, **kwargs):
    model_args = dict(
        block=SEResNeXtBottleneck, layers=[3, 4, 23, 3], groups=32, reduction=16, **kwargs)
    return _create_senet(num_classes, 'cifar_seresnext101_32x4d', **model_args)
