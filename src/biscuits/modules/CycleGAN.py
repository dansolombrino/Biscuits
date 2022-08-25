import logging
from glob import glob
from telnetlib import BM
from typing import Mapping

import hydra
import omegaconf
from nn_core.common import PROJECT_ROOT


import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

pylogger = logging.getLogger(__name__)

from typing import Sequence, List
import random
from torchsummary import summary



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

    elif init_method == "normal":
        init.normal_(layer.weight, init_range[0], init_range[1])

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


def _freeze_layer_params(layer: nn.Module, should_freeze_parameters: bool):

    if should_freeze_parameters:
        for p in layer.parameters(recurse=False):
            p.requires_grad = False


class ResidualBlock(nn.Module):
    def __init__(
        self, 
        in_features: int,
        conv_init: omegaconf.DictConfig,
        instancenorm_init: omegaconf.DictConfig,
        conv_should_freeze_parameters: bool,
        instancenorm_should_freeze_parameters: bool
    ) -> None:
        """
        A generic residual block.

        The input is transformed by a block,
        then, the transformation is summed up to the original input

        :param in_features: number of input features
        """
        super().__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor with shape [batch, channels, w, h]

        :returns: tensor with shape [batch, channels, w, h]
        """
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(
        self, 
        input_shape: Sequence[int], 
        num_residual_blocks: int,
        conv_init: omegaconf.DictConfig,
        instancenorm_init: omegaconf.DictConfig,
        conv_should_freeze_parameters: bool,
        instancenorm_should_freeze_parameters: bool
    ) -> None:
        """
        Image-conditioned image generator.

        It takes in input an image and produces another image.

        :param input_shape: shape of expected input image
        :param num_residual_blocks: number of residual blocks to use
        """
        super().__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features, affine=True),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [
                ResidualBlock(
                    in_features=out_features,
                    conv_init = conv_init,
                    instancenorm_init = instancenorm_init,
                    conv_should_freeze_parameters = 
                        conv_should_freeze_parameters,
                    instancenorm_should_freeze_parameters = 
                        instancenorm_should_freeze_parameters 
                )
            ]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features, affine=True),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(out_features, channels, 7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

        for b in self.model:
            if "Conv" in b.__class__.__name__:
                _init_layer_params(
                    layer=b,
                    init_method=conv_init.method,
                    init_range=conv_init.range
                )
                
                _freeze_layer_params(
                    layer=b, 
                    should_freeze_parameters=conv_should_freeze_parameters
                )

            if "InstanceNorm" in b.__class__.__name__:
                _init_layer_params(
                    layer=b,
                    init_method=instancenorm_init.parameters.method,
                    init_range=instancenorm_init.parameters.range
                )

                _init_layer_bias(
                    layer=b,
                    init_method=instancenorm_init.bias
                )

                _freeze_layer_params(
                    layer=b, 
                    should_freeze_parameters=
                        instancenorm_should_freeze_parameters
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: tensor with shape [batch, channels, w, h]

        :returns: tensor with shape [batch, channels, w, h]
        """
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(
        self, 
        input_shape: Sequence[int],
        conv_init: omegaconf.DictConfig,
        instancenorm_init: omegaconf.DictConfig,
        conv_should_freeze_parameters: bool,
        instancenorm_should_freeze_parameters: bool
    ) -> None:
        """
        Discriminator that tries to infer if an image is:
        - fake, i.e. it has been generated by a generator
        - real, i.e. it has not been generated by a generator

        :param input_shape: shape of the expected image
        """
        super().__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(
            in_filters: int, out_filters: int, normalize: bool = True
        ) -> Sequence[nn.Module]:
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

        for b in self.model:
            if "Conv" in b.__class__.__name__:
                _init_layer_params(
                    layer=b,
                    init_method=conv_init.method,
                    init_range=conv_init.range
                )
                
                _freeze_layer_params(
                    layer=b, 
                    should_freeze_parameters=conv_should_freeze_parameters
                )

            if "InstanceNorm" in b.__class__.__name__:
                _init_layer_params(
                    layer=b,
                    init_method=instancenorm_init.parameters.method,
                    init_range=instancenorm_init.parameters.range
                )

                _init_layer_bias(
                    layer=b,
                    init_method=instancenorm_init.bias
                )

                _freeze_layer_params(
                    layer=b, 
                    should_freeze_parameters=
                        instancenorm_should_freeze_parameters
                )

    def forward(self, img: torch.tensor) -> torch.Tensor:
        """
        :param img: tensor with shape [batch, channels, w, h]

        :returns: tensor with shape [batch, 1, 3, 3]
        """
        return self.model(img)


class ReplayBuffer:
    def __init__(self, max_size: int = 50) -> None:
        """
        Image buffer to increase the robustness of the generator.

        Once it is full, i.e. it contains max_size images, each image in a given batch
        is swapped with probability p=0.5 with another one contained in the buffer.

        """
        assert (
            max_size > 0
        ), "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fill the buffer with each element in data.
        If the buffer is full, with p=0.5 swap each element in data with
        another in the buffer.

        :param data: tensor with shape [batch, ...]

        :returns: tensor with shape [batch, ...]
        """
        to_return = []

        for i in range(data.shape[0]):
            element = data[[i], ...]

            if len(self.data) < self.max_size:
                self.data.append(element)

            elif random.uniform(0, 1) > 0.5:
                i = random.randint(0, self.max_size - 1)
                self.data[i], element = element, self.data[i]

            to_return.append(element)

        return torch.cat(to_return)


class LambdaLR:
    def __init__(self, n_epochs: int, decay_start_epoch: int) -> None:
        """
        Linearly decay the leraning rate to 0, starting from `decay_start_epoch`
        to the final epoch.

        In practice

        :param n_epochs: total number of epochs
        :param decay_start_epoch: epoch in which the learning rate starts to decay
        """
        assert (
            n_epochs - decay_start_epoch
        ) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch: int) -> float:
        """
        One step of lr decay:
        - if `epoch < self.decay_start_epoch` it doesn't change the learning rate.
        - Otherwise, it linearly decay the lr to reach zero

        :param epoch: current epoch
        :returns: learning rate multiplicative factor
        """
        return 1.0 - max(0, epoch - self.decay_start_epoch) / (
            self.n_epochs - self.decay_start_epoch
        )



def _debug_CycleGAN(
    conv_init: omegaconf.DictConfig,
    instancenorm_init: omegaconf.DictConfig,
    conv_should_freeze_parameters: bool,
    instancenorm_should_freeze_parameters: bool
):

    ### internal residual block

    res = ResidualBlock(
        in_features=3,
        conv_init = conv_init,
        instancenorm_init = instancenorm_init,
        conv_should_freeze_parameters = conv_should_freeze_parameters,
        instancenorm_should_freeze_parameters = instancenorm_should_freeze_parameters 
    )

    batch = torch.rand(10, 3, 128, 128).cuda()

    summary(res.cuda(), (3, 128, 128))
    print(res(batch).shape)

    # generator

    g = GeneratorResNet(
        input_shape=[3, 50, 50], 
        num_residual_blocks=6, # potential entry point for dynamic selection of resnet depth
        conv_init = conv_init,
        instancenorm_init = instancenorm_init,
        conv_should_freeze_parameters = conv_should_freeze_parameters,
        instancenorm_should_freeze_parameters = instancenorm_should_freeze_parameters 
    ).cuda()

    summary(g, (3, 128, 128))
    print(g(batch).shape)

    # discriminator 

    d = Discriminator(
        input_shape=[3, 50, 50],
        conv_init = conv_init,
        instancenorm_init = instancenorm_init,
        conv_should_freeze_parameters = conv_should_freeze_parameters,
        instancenorm_should_freeze_parameters = instancenorm_should_freeze_parameters 
    ).cuda()

    batch = torch.rand(2, 3, 50, 50).cuda()

    summary(d, (3, 50, 50))
    print(d(batch).shape)

    # image buffer

    b = ReplayBuffer(max_size=5)
    batch_s = 0
    batch_size = 5
    batch_e = batch_s + batch_size

    a = torch.arange(batch_s, batch_e)[..., None]
    batch_s = batch_e
    batch_e = batch_s + batch_size
    batch = b.push_and_pop(a)
    print(
        f"Input batch:\n{a}\n\nOutput batch:\n{batch}\n\nHidden buffer state:\n{b.data}"
    )

    # LR schedule
    n_epochs = 10
    decay_from = 3
    lr = LambdaLR(n_epochs, decay_from)
    for i in range(n_epochs + 1):
        if i == decay_from:
            print("\tStarting to decay")
        print(lr.step(i))
        if i == n_epochs:
            print("\tEnd of the decay")



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    
    _debug_CycleGAN(
        conv_init=cfg.nn.model.conv_init,
        instancenorm_init=cfg.nn.model.instancenorm_init,
        conv_should_freeze_parameters=cfg.nn.model.conv_freeze_parameters,
        instancenorm_should_freeze_parameters=cfg.nn.model.instancenorm_freeze_parameters,
    )

if __name__ == "__main__":

    main()
