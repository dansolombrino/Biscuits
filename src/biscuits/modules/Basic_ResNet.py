from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

__all__ = [
    "ResNet",
    "resnet14",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnet1202",
]


def _get_num_parameters(net: nn.Module, trainable: bool) -> bool:
    import numpy as np

    total_params = 0

    param_list = net.parameters()
    if trainable is not None:
        param_list = filter(lambda p: p.requires_grad == trainable, param_list)

    for x in param_list:
        total_params += np.prod(x.data.numpy().shape)

    return total_params


def get_num_trainable_parameters(net):
    return _get_num_parameters(net=net, trainable=True)


def get_num_not_trainable_parameters(net):
    return _get_num_parameters(net=net, trainable=False)


def get_num_parameters(net):
    return _get_num_parameters(net=net, trainable=None)


def get_num_layers(net):
    return len(
        list(
            filter(
                lambda p: p.requires_grad and len(p.data.size()) > 1,
                net.parameters(),
            )
        )
    )


def print_num_summary(net):
    total_params = get_num_parameters(net)
    total_trainable_params = get_num_trainable_parameters(net)
    total_not_trainable_params = get_num_not_trainable_parameters(net)
    print("Total number of params              ", total_params)
    print("Total number of trainable params    ", total_trainable_params)
    print("Total number of NOT trainable params", total_not_trainable_params)

    total_layers = get_num_layers(net)
    print("Total layers                        ", total_layers)


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

    if isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight, 0, 1)
        init.constant_(m.bias, 0)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="A"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)

        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)

        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet14():
    return ResNet(BasicBlock, [2, 2, 2])


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def ResNetFactory(depth: int) -> ResNet:
    return globals()["resnet" + str(depth)]()


def test(net):

    print_num_summary(net)


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)

            net = globals()[net_name]()
            # test(net)

            print()
