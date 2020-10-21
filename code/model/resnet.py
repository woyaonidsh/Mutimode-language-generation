import os
import torch.nn as nn
import torchvision.models.resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
import torch.utils.model_zoo as model_zoo

model_urls = \
    {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        for i in range(2, 5):
            getattr(self, 'layer%d' % i)[0].conv1.stride = (2, 2)
            getattr(self, 'layer%d' % i)[0].conv2.stride = (1, 1)


def resnet18(save_path, pretrained=False, image_embed =1000):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2] , image_embed)
    if pretrained:
        if pretrained:
            if os.path.exists(path=save_path):
                model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=save_path))
            else:
                os.mkdir(save_path)
                model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir=save_path))
    return model


def resnet34(save_path, pretrained=False, image_embed =1000):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], image_embed)
    if pretrained:
        if pretrained:
            if os.path.exists(path=save_path):
                model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=save_path))
            else:
                os.mkdir(save_path)
                model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir=save_path))
    return model


def resnet50(save_path, pretrained=False, image_embed =1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], image_embed)
    if pretrained:
        if pretrained:
            if os.path.exists(path=save_path):
                model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=save_path))
            else:
                os.mkdir(save_path)
                model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir=save_path))
    return model


def resnet101(save_path, pretrained=False, image_embed =1000):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], image_embed)
    if pretrained:
        if os.path.exists(path=save_path):
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=save_path))
        else:
            os.mkdir(save_path)
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir=save_path))
    return model


def resnet152(save_path, pretrained=False, image_embed =1000):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], image_embed)
    if pretrained:
        if os.path.exists(path=save_path):
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir=save_path))
        else:
            os.mkdir(save_path)
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir=save_path))
    return model
