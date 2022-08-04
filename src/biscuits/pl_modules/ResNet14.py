from turtle import forward

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

EXPANSION = 1


def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


class block(nn.Module):
    def __init__(
        self,
        in_channels,
        intermediate_channels,
        identity_downsample=None,
        stride=1,
    ):
        super(block, self).__init__()

        self.expansion = EXPANSION

        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        # self.conv3 = nn.Conv2d(
        #     intermediate_channels,
        #     intermediate_channels * self.expansion,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False
        # )
        # self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # x = self.conv3(x)
        # x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )

        # self.layer4 = self._make_layer(
        #     block, layers[3], intermediate_channels=512, stride=2
        # )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * EXPANSION, num_classes)
        self.fc = nn.Linear(256 * EXPANSION, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(
        self, block, num_residual_blocks, intermediate_channels, stride
    ):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * EXPANSION:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * EXPANSION,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * EXPANSION),
            )

        layers.append(
            block(
                self.in_channels,
                intermediate_channels,
                identity_downsample,
                stride,
            )
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * EXPANSION

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    # return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)
    return ResNet(block, [6, 6, 6], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)


def test():
    DESIRED_RESNET_DEPTH = 110
    num_res_blocks_to_reach_desired_depth = (DESIRED_RESNET_DEPTH - 2) / 6
    print(num_res_blocks_to_reach_desired_depth)
    net = ResNet(block, [18, 18, 18], image_channels=3, num_classes=1000)

    y = net(torch.randn(4, 3, 224, 224)).to("cuda")

    # pprint(str(net))
    import re
    from pprint import pprint

    l = [m.start() for m in re.finditer("\(conv", str(net))]
    print(len(l))

    if len(l) + 1 != DESIRED_RESNET_DEPTH:
        print(
            f"ERROR! Specified params did NOT yield to the creation of a ResNet-{DESIRED_RESNET_DEPTH}"
        )
        print(f"       DESIRED_RESNET_DEPTH: {DESIRED_RESNET_DEPTH}")
        print(f"       Resulting NN depth  : {len(l) + 1}")
    else:
        print(f"ResNet-{DESIRED_RESNET_DEPTH} succesfully created!")


test()
