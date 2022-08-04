import re
from turtle import forward

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

EXPANSION = 1


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

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * EXPANSION, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

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


def test():
    DESIRED_RESNET_DEPTH = 110

    # number of res blocks (in each of the 3 layers!) needed to reach desired depth
    num_res_blocks = int((DESIRED_RESNET_DEPTH - 2) / 6)

    net = ResNet(
        block,
        [num_res_blocks, num_res_blocks, num_res_blocks],
        image_channels=3,
        num_classes=10,
    )

    y = net(torch.randn(4, 3, 224, 224)).to("cuda")

    num_conv_occurrencies = len(
        [m.start() for m in re.finditer("\(conv", str(net))]
    )

    if num_conv_occurrencies + 1 != DESIRED_RESNET_DEPTH:
        print(
            f"ERROR! Specified params did NOT yield to the creation of a ResNet-{DESIRED_RESNET_DEPTH}"
        )
        print(f"       DESIRED_RESNET_DEPTH: {DESIRED_RESNET_DEPTH}")
        print(f"       Resulting NN depth  : {num_conv_occurrencies + 1}")
        exit()
    else:
        print(f"ResNet-{DESIRED_RESNET_DEPTH} succesfully created!")

    tot_params = sum(p.numel() for p in net.parameters())
    print(tot_params)


test()
