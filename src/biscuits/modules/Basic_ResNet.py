from glob import glob
from typing import Mapping

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import logging

pylogger = logging.getLogger(__name__)

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

NUM_CLASSES = 10
FREEZE_CONV_PARAMS = False
FREEZE_BATCHNORM_PARAMS = False


def _count_num_parameters(param_list, trainable):
    import numpy as np

    total_params = 0

    if trainable is not None:
        param_list = filter(lambda p: p.requires_grad == trainable, param_list)

    for x in param_list:
        total_params += np.prod(torch.clone(x).cpu().data.numpy().shape)

    return total_params


def _get_num_parameters(net: nn.Module, trainable: bool) -> bool:

    param_list = net.parameters()

    return _count_num_parameters(param_list, trainable)


def get_num_trainable_parameters(net):
    return _get_num_parameters(net=net, trainable=True)


def get_num_not_trainable_parameters(net):
    return _get_num_parameters(net=net, trainable=False)


def get_num_parameters(net):
    return _get_num_parameters(net=net, trainable=None)


# does NOT distinguish between trainable and NON-trainable layers, for now
def get_num_layers(net):
    return len(
        list(
            filter(
                # lambda p: p.requires_grad and len(p.data.size()) > 1,
                lambda p: p.requires_grad and len(p.data.size()) > 1,
                net.parameters(),
            )
        )
    )

def compute_num_summary(net):
    total_params = get_num_parameters(net)
    total_trainable_params = get_num_trainable_parameters(net)
    total_not_trainable_params = get_num_not_trainable_parameters(net)
    total_layers = get_num_layers(net)

    return f"Total layers                        , {total_layers}\n" + \
           f"Total number of params              , {total_params}\n" + \
           f"Total number of trainable params    , {total_trainable_params}\n" + \
           f"Total number of NOT trainable params, {total_not_trainable_params}\n"

def print_num_summary(net):
    print(compute_num_summary(net))
    


def _weights_init(m, conv_init_method: str, batchnorm_init_methods):
    classname = m.__class__.__name__
    # print(classname)

    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)

    if isinstance(m, nn.Conv2d):
        # switch case on conv_init_method goes here
        init.kaiming_normal_(m.weight)

    if isinstance(m, nn.BatchNorm2d):
        # switch case on batchnorm_init_methods goes here
        init.uniform_(m.weight, 0, 1)
        init.constant_(m.bias, 0)


class LambdaLayer(nn.Module):
# class LambdaLayer(pl.LightningModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
# class BasicBlock(pl.LightningModule):
    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        conv_init_method: str,
        batchnorm_init_methods: Mapping,
        stride=1,
        option="A",
    ):
        super(BasicBlock, self).__init__()

        # --- begin 1st Convolutional layer of basic Residual Block
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


def _init_layer_params(layer, init_method, init_range=None):

    # using this instead of switch-case to ensure Python < 3.10 compatibility
    if init_method.isnumeric():
        init.constant_(layer.weight, float(init_method))

    elif init_method == "he_kaiming_normal":
        init.kaiming_normal_(layer.weight)

    elif init_method == "uniform":
        if init_range is None:
            pylogger.warning(
                "Requested uniform init, but sampling range not given... "
                "defaulting to [0, 1] range"
            )

            init_range = [0, 1]

        init.uniform_(layer.weight, init_range[0], init_range[1])

    # default case
    else:
        init.kaiming_normal_(layer.weight)


def _init_layer_bias(layer, init_method, init_range=None):

    # using this instead of switch-case to ensure Python < 3.10 compatibility
    if init_method.isnumeric():
        init.constant_(layer.bias, float(init_method))

    # default case
    else:
        init.constant_(layer.bias, float(init_method))

class ResNet(nn.Module):
# class ResNet(pl.LightningModule):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes,
        conv_freeze_params: bool,
        batchnorm_freeze_params: bool,
        conv_init_method: str,
        batchnorm_init_methods: Mapping,
        lin_init_method: str,
    ):
        self.conv_init_method = conv_init_method
        self.batchnorm_init_methods = batchnorm_init_methods

        super(ResNet, self).__init__()
        self.in_planes = 16

        # --- begin 1st NN block (classic Convolutional w/ batch norm) ---

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        _init_layer_params(self.conv1, conv_init_method)

        self.bn1 = nn.BatchNorm2d(16)
        _init_layer_params(self.bn1, batchnorm_init_methods["parameters"])
        _init_layer_bias(self.bn1, batchnorm_init_methods["bias"])

        # --- end 1st NN block (classic Convolutional w/ batch norm) ---

        # --- begin 2nd NN block (1st residual Convolutional w/ batch norm)

        self.layer1 = self._create_residual_layer(
            block=block,
            planes=16,
            num_blocks=num_blocks[0],
            stride=1,
            conv_init_method=conv_init_method,
            batchnorm_init_methods=batchnorm_init_methods,
        )

        # --- end 2nd NN block (1st residual Convolutional w/ batch norm)

        # --- begin 3rd NN block (2nd residual Convolutional w/ batch norm)
        self.layer2 = self._create_residual_layer(
            block=block,
            planes=32,
            num_blocks=num_blocks[1],
            stride=2,
            conv_init_method=conv_init_method,
            batchnorm_init_methods=batchnorm_init_methods,
        )
        # --- end 3rd NN block (2nd residual Convolutional w/ batch norm)

        # --- begin 4rd NN block (3rd residual Convolutional w/ batch norm)
        self.layer3 = self._create_residual_layer(
            block=block,
            planes=64,
            num_blocks=num_blocks[2],
            stride=2,
            conv_init_method=conv_init_method,
            batchnorm_init_methods=batchnorm_init_methods,
        )
        # --- begin 4rd NN block (3rd residual Convolutional w/ batch norm)

        self.linear = nn.Linear(64, num_classes)
        _init_layer_params(self.linear, lin_init_method)

    def _create_residual_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        conv_init_method: str,
        batchnorm_init_methods: Mapping,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes,
                    planes=planes,
                    conv_init_method="he_kaiming_normal",
                    batchnorm_init_methods={},
                    stride=stride,
                )
            )
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        if FREEZE_CONV_PARAMS:
            for p in self.conv1.parameters():
                p.requires_grad = False

        if FREEZE_BATCHNORM_PARAMS:
            for p in self.bn1.parameters():
                p.requires_grad = False

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet14(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [2, 2, 2],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def resnet20(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [3, 3, 3],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def resnet32(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [5, 5, 5],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def resnet44(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [7, 7, 7],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def resnet56(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [9, 9, 9],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def resnet110(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [18, 18, 18],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def resnet1202(
    conv_init_method: str, batchnorm_init_methods: Mapping, lin_init_method: str
):
    return ResNet(
        BasicBlock,
        [200, 200, 200],
        NUM_CLASSES,
        FREEZE_CONV_PARAMS,
        FREEZE_BATCHNORM_PARAMS,
        conv_init_method,
        batchnorm_init_methods,
        lin_init_method,
    )


def ResNetFactory(
    depth: int,
    conv_init_method: str,
    batchnorm_init_methods: Mapping,
    lin_init_method: str,
) -> ResNet:
    return globals()["resnet" + str(depth)](
        conv_init_method=conv_init_method,
        batchnorm_init_methods=batchnorm_init_methods,
        lin_init_method=lin_init_method,
    )


def test(net):

    print_num_summary(net)


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith("resnet"):
            print(net_name)

            net = globals()[net_name]()
            # test(net)

            print()
