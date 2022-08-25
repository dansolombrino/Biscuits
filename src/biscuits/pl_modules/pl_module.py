import logging
from pickletools import optimize
from typing import Any, Mapping, Optional, Sequence, Tuple, Union, Dict, List


import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from pathlib import Path
from dataclasses import dataclass
from torchvision import transforms
from PIL import Image
from torch import optim
import itertools
from torch import nn
import wandb
import torchvision

from biscuits.data.datamodule import MetaData
from biscuits.modules import Advanced_ResNet, Basic_ResNet, CycleGAN
from biscuits.modules.CycleGAN import GeneratorResNet
from biscuits.modules.CycleGAN import Discriminator
from biscuits.modules.CycleGAN import ReplayBuffer
from biscuits.modules.CycleGAN import LambdaLR
from biscuits.modules.Advanced_ResNet import (
    Advanced_ResNet,
    compute_num_summary,
)
from biscuits.modules.module import CNN

pylogger = logging.getLogger(__name__)


# class MyLightningModule(pl.LightningModule):
class BasicResNetLightningModule(pl.LightningModule):

    logger: NNLogger

    def __init__(
        self, metadata: Optional[MetaData] = None, *args, **kwargs
    ) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        # self.metadata = metadata

        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        self.resnet_depth = kwargs["resnet_depth"]

        self.conv_init_method = kwargs["conv_init_method"]
        self.batchnorm_init_methods = kwargs["batchnorm_init_methods"]
        self.lin_init_method = kwargs["lin_init_method"]

        self.conv_freeze_parameters = kwargs["conv_freeze_parameters"]
        self.batchnorm_freeze_parameters = kwargs["batchnorm_freeze_parameters"]
        self.lin_freeze_parameters = kwargs["lin_freeze_parameters"]

        self.dropout_probability = kwargs["dropout_probability"]
        self.dropout2d_probability = kwargs["dropout2d_probability"]

        self.optimizer = kwargs["optimizer"]
        try:
            self.lr_scheduler = kwargs["lr_scheduler"]
        except KeyError:
            self.lr_scheduler = None

        self.model = Basic_ResNet.ResNetFactory(
            self.resnet_depth,
            self.conv_init_method,
            self.batchnorm_init_methods,
            self.lin_init_method,
            self.conv_freeze_parameters,
            self.batchnorm_freeze_parameters,
            self.lin_freeze_parameters,
            self.dropout_probability,
            self.dropout2d_probability,
        )

        pylogger.info("Instantiated model: ")
        pylogger.info(Basic_ResNet.compute_num_summary(self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(x)

    def step(self, x, y) -> Mapping[str, Any]:

        logits = self(x)

        loss = F.cross_entropy(logits, y)

        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        # if "lr_scheduler" not in kwargs:
        if self.lr_scheduler is None:
            return [opt]

        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


def _debug_BasicResNetLighningModule(cfg: omegaconf.DictConfig):

    model: pl.LightningModule = hydra.utils.instantiate(
        config=cfg.nn.module,
        optimizer=cfg.nn.model.optimizer,
        resnet_depth=cfg.nn.model.resnet_depth,
        conv_init_method=cfg.nn.model.conv_init_method,
        batchnorm_init_methods=cfg.nn.model.batchnorm_init_methods,
        lin_init_method=cfg.nn.model.lin_init_method,
        conv_freeze_parameters=cfg.nn.model.conv_freeze_parameters,
        batchnorm_freeze_parameters=cfg.nn.model.batchnorm_freeze_parameters,
        lin_freeze_parameters=cfg.nn.model.lin_freeze_parameters,
        dropout_probability=cfg.nn.model.dropout_probability,
        dropout2d_probability=cfg.nn.model.dropout2d_probability,
        _recursive_=False,
    )

    print(model)


class AdvancedResNetLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(
        self, metadata: Optional[MetaData] = None, *args, **kwargs
    ) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        # self.metadata = metadata

        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        self.resnet_depth = kwargs["resnet_depth"]

        self.lin_init_method = kwargs["lin_init_method"]
        self.lin_freeze_parameters = kwargs["lin_freeze_parameters"]

        self.dropout_probability = kwargs["dropout_probability"]

        self.transfer_learning = kwargs["transfer_learning"]
        
        self.in_channels = kwargs["in_channels"]

        self.num_classes = kwargs["num_classes"]

        self.optimizer = kwargs["optimizer"]
        try:
            self.lr_scheduler = kwargs["lr_scheduler"]
        except KeyError:
            self.lr_scheduler = None

        self.model = Advanced_ResNet.from_pretrained(
            model_name="resnet" + str(self.resnet_depth),
            num_classes=self.num_classes,
            lin_init_method=self.lin_init_method,
            lin_freeze_parameters=self.lin_freeze_parameters,
            transfer_learning=self.transfer_learning,
            dropout_probability=self.dropout_probability,
            in_channels=self.in_channels
        )

        pylogger.info("Instantiated model: ")
        pylogger.info(compute_num_summary(self.model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(x)

    def step(self, x, y) -> Mapping[str, Any]:

        logits = self(x)

        loss = F.cross_entropy(logits, y)

        return {"logits": logits.detach(), "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # example
        x, y = batch
        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        # if "lr_scheduler" not in kwargs:
        if self.lr_scheduler is None:
            return [opt]

        scheduler = hydra.utils.instantiate(self.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]    

def _debug_AdvancedResNetLighningModule(cfg: omegaconf.DictConfig):

    model: pl.LightningModule = hydra.utils.instantiate(
        config=cfg.nn.module,
        optimizer=cfg.nn.model.optimizer,
        resnet_depth=cfg.nn.model.resnet_depth,
        # conv_init_method=cfg.nn.model.conv_init_method,
        # batchnorm_init_methods=cfg.nn.model.batchnorm_init_methods,
        lin_init_method=cfg.nn.model.lin_init_method,
        # conv_freeze_parameters=cfg.nn.model.conv_freeze_parameters,
        # batchnorm_freeze_parameters=cfg.nn.model.batchnorm_freeze_parameters,
        lin_freeze_parameters=cfg.nn.model.lin_freeze_parameters,
        dropout_probability=cfg.nn.model.dropout_probability,
        # dropout2d_probability=cfg.nn.model.dropout2d_probability,
        transfer_learning=cfg.nn.model.transfer_learning,
        # in_channels=cfg.nn.model.in_channels,
        in_channels=cfg.data.datasets.in_channels,
        _recursive_=False,
    )

    print(model)


class CycleGANLightningModule(pl.LightningModule):
    def __init__(
        # self,
        # hparams: Union[Dict, Config],
        # trainA_folder: Path,
        # trainB_folder: Path,
        # testA_folder: Path,
        # testB_folder: Path,
        self, metadata: Optional[MetaData] = None, *args, **kwargs
    ) -> None:
        """
        The CycleGAN model.

        :param hparams: dictionary that contains all the hyperparameters
        :param trainA_folder: Path to the folder that contains the trainA images
        :param trainB_folder: Path to the folder that contains the trainB images
        :param testA_folder: Path to the folder that contains the testA images
        :param testB_folder: Path to the folder that contains the testB images
        """
        super().__init__()
        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))


        # Dataset paths
        # self.trainA_folder = trainA_folder
        # self.trainB_folder = trainB_folder
        # self.testA_folder = testA_folder
        # self.testB_folder = testB_folder

        # Expected image shape
        # self.input_shape = (self.hparams["channels"], self.hparams["img_height"], self.hparams["img_width"])
        self.input_shape = tuple(
            map(
                int, 
                kwargs[
                    "input_shape"
                ].replace('(', '').replace(')', '').split(', ')
            )
        )

        # Generators A->B and B->A
        # self.G_AB = GeneratorResNet(self.input_shape, self.hparams["n_residual_blocks"])
        # self.G_BA = GeneratorResNet(self.input_shape, self.hparams["n_residual_blocks"])
        self.G_AB = GeneratorResNet(
            input_shape=self.input_shape, 
            num_residual_blocks=kwargs["num_residual_blocks"],
            conv_init=kwargs["conv_init"],
            instancenorm_init=kwargs["instancenorm_init"],
            conv_should_freeze_parameters=kwargs["conv_freeze_parameters"],
            instancenorm_should_freeze_parameters=kwargs["instancenorm_freeze_parameters"]
        )
        self.G_BA = GeneratorResNet(
            input_shape=self.input_shape, 
            num_residual_blocks=kwargs["num_residual_blocks"],
            conv_init=kwargs["conv_init"],
            instancenorm_init=kwargs["instancenorm_init"],
            conv_should_freeze_parameters=kwargs["conv_freeze_parameters"],
            instancenorm_should_freeze_parameters=kwargs["instancenorm_freeze_parameters"]
        )

        # Discriminators
        self.D_A = Discriminator(
            input_shape=self.input_shape,
            conv_init=kwargs["conv_init"],
            instancenorm_init=kwargs["instancenorm_init"],
            conv_should_freeze_parameters=kwargs["conv_freeze_parameters"],
            instancenorm_should_freeze_parameters=kwargs["instancenorm_freeze_parameters"]
        )
        self.D_B = Discriminator(
            input_shape=self.input_shape,
            conv_init=kwargs["conv_init"],
            instancenorm_init=kwargs["instancenorm_init"],
            conv_should_freeze_parameters=kwargs["conv_freeze_parameters"],
            instancenorm_should_freeze_parameters=kwargs["instancenorm_freeze_parameters"]
        )

        # Initialize weights
        # https://pytorch.org/docs/stable/nn.html?highlight=nn%20module%20apply#torch.nn.Module.apply
        # self.G_AB.apply(self.weights_init_normal)
        # self.G_BA.apply(self.weights_init_normal)
        # self.D_A.apply(self.weights_init_normal)
        # self.D_B.apply(self.weights_init_normal)

        # Image Normalizations
        # self.image_transforms = transforms.Compose(
        #     [
        #         transforms.Resize(int(self.hparams["img_height"] * 1.12), Image.BICUBIC),
        #         transforms.RandomCrop((self.hparams["img_height"], self.hparams["img_width"])),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )

        # Image Normalization for the validation: remove source of randomness
        # self.val_image_transforms = transforms.Compose(
        #     [
        #         transforms.Resize(int(self.hparams["img_height"] * 1.12), Image.BICUBIC),
        #         transforms.CenterCrop((self.hparams["img_height"], self.hparams["img_width"])),
        #         # transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )

        # Image buffers
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Forward pass cache to avoid re-doing some computation
        self.fake_A = None
        self.fake_B = None

        # Losses
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()

        # Ignore this.
        # It avoids wandb logging when lighting does a sanity check on the validation
        self.is_sanity = True

        self.lr = kwargs["lr"]
        self.beta_1 = kwargs["beta_1"]
        self.beta_2 = kwargs["beta_2"]
        self.num_epochs = kwargs["num_epochs"],
        self.decay_epoch = kwargs["decay_epoch"]


    def forward(self, x: torch.Tensor, a_to_b: bool) -> torch.Tensor:
        """
        Forward pass for this model.

        This is not used while training!

        :param x: input of the forward pass with shape [batch, channel, w, h]
        :param a_to_b: if True uses the mapping A->B, otherwise uses B->A

        :returns: the translated image with shape [batch, channel, w, h]
        """
        if a_to_b:
            return self.G_AB(x)
        else:
            return self.G_BA(x)


    def configure_optimizers(
        self,
    ) -> Tuple[Sequence[optim.Optimizer], Sequence[Dict[str, Any]]]:
        """ Instantiate the optimizers and schedulers.

        We have three optimizers (and relative schedulers):

        - Optimizer with index 0: optimizes the parameters of both generators
        - Optimizer with index 1: optimizes the parameters of D_A
        - Optimizer with index 2: optimizes the parameters of D_B

        Each scheduler implements a linear decay to 0 after `cfg.hparams["decay_epoch"]`

        :returns: the optimizers and relative schedulers (look at the return type!)
        """
        # Optimizers
        # TODO instanciate optims via hydra, if possible and practical
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=self.lr,
            betas=(self.beta_1, self.beta_2),
        )
        optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2)
        )
        optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2)
        )

        # Schedulers for each optimizers
        # TODO instanciate schedulers via hydra, if possible and practical
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            optimizer_G,
            lr_lambda=LambdaLR(self.num_epochs, self.decay_epoch).step,
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_A,
            lr_lambda=LambdaLR(self.num_epochs, self.decay_epoch).step,
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            optimizer_D_B,
            lr_lambda=LambdaLR(self.num_epochs, self.decay_epoch).step,
        )

        return (
            [optimizer_G, optimizer_D_A, optimizer_D_B],
            [
                {"scheduler": lr_scheduler_G, "interval": "epoch", "frequency": 1},
                {"scheduler": lr_scheduler_D_A, "interval": "epoch", "frequency": 1},
                {"scheduler": lr_scheduler_D_B, "interval": "epoch", "frequency": 1},
            ],
        )


    def criterion_GAN(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ The loss criterion for GAN losses

        :param x: tensor with any shape
        :param y: tensor with any shape
        
        :returns: the mse between x and y
        """
        return self.mse(x, y)

    def criterion_cycle(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ The loss criterion for Cycle losses

        :param x: tensor with any shape
        :param y: tensor with any shape
        
        :returns: the l1 between x and y
        """
        return self.l1(x, y)

    def criterion_identity(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """ The loss criterion for Identity losses

        :param x: tensor with any shape
        :param y: tensor with any shape
        
        :returns: the l1 between x and y
        """
        return self.l1(x, y)

    def identity_loss(self, image: torch.Tensor, generator: nn.Module) -> torch.Tensor:
        """ Implements the identity loss for the given generator

        :param generator: a generator module that maps X -> Y
        :param image: an image in the Y distribution with shape [batch, channel, w, h]

        :returns: the identity loss for these (generator, image)
        """
        return self.criterion_identity(generator(image), image)

    def gan_loss(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        image: torch.Tensor,
        expected_label: torch.Tensor,
    ) -> torch.Tensor:
        """ Implements the GAN loss for the given generator and discriminator

        :param image: the input image with shape [batch, channle, w, h]
        :param generator: the generator module to use to translate the image from X -> Y
        :param discriminator: the discriminator that tries to distinguish fake and real images
        :expected_label: tensor with shape compatible to the discriminator's output.
                        It is full of ones when training the generator. We feed a fake 
                        image to the discriminator and we expect to get ones 
                        (for the discriminator this is an error!)
        
        :returns: the GAN loss for these (image, generator, discriminator)
        """
        fake_image = generator(image)
        predicted_label = discriminator(fake_image)
        loss_GAN = self.criterion_GAN(predicted_label, expected_label)
        return loss_GAN, fake_image

    def cycle_loss(
        self,
        fake_image: torch.Tensor,
        reverse_generator: nn.Module,
        original_image: torch.Tensor,
    ) -> torch.Tensor:
        """ Implements the cycle consistency loss

        It takes in input a fake image, to avoid repeated computation, 
        thus it only needs the reverse mapping that produced that fake image.

        :param fake_image: a image produced by a mapping X->Y with shape [batch, channel, w, h]
        :param reverse_generator: the generator module that maps Y->X
        :param original_image: the original image in X with shape [batch, channel, w, h] 
                            to compare with the reconstructed fake image
        
        :returns: the cycle consistency loss for this (fake_image, reverse_generator, original_image)
        """ 
        recovered_image = reverse_generator(fake_image)
        return self.criterion_cycle(recovered_image, original_image)

    def discriminator_loss(
        self,
        discriminator: nn.Module,
        proposed_image: torch.Tensor,
        expected_label: torch.Tensor,
    ) -> torch.Tensor:
        """ Implements the loss used to train the discriminator

        :param discriminator: the discriminator model to train
        :param proposed_image: the fake or real image proposed with shape [batch, channel, w, h]
        :param expected_label: tensor with shape compatible to the discriminator's output, 
                            full of zeros if the proposed image is fake
                            full of ones if the proposed image is real
        
        :returns: the discriminator loss for this (discriminator, proposed_image, expected_label)
        """
        predicted_label = discriminator(proposed_image)
        return self.criterion_GAN(predicted_label, expected_label)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_nb: int, optimizer_idx: int
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """ Implements a single training step

        The parameter `optimizer_idx` identifies with optimizer "called" this training step,
        this we can change the behaviour of the training depending on which optimizers
        is currently performing the optimization

        :param batch: current training batch
        :param batch_nb: the index of the current batch
        :param optimizer_idx: the index of the optimizer in use, see the function `configure_optimizers`

        :returns: the total loss for the current training step, together with other information for the
                  logging and possibly the progress bar
        """
        # Unpack the batch
        real_A = batch["A"]
        real_B = batch["B"]

        # Adversarial ground truths
        valid = torch.ones(
            (real_A.size(0), *self.D_A.output_shape), device=real_A.device
        )
        fake = torch.zeros(
            (real_A.size(0), *self.D_A.output_shape), device=real_A.device
        )

        # The first optimizer is for the two generators!
        if optimizer_idx == 0:

            # Identity A and B loss
            loss_id_A = self.identity_loss(real_A, self.G_BA)
            loss_id_B = self.identity_loss(real_B, self.G_AB)
            loss_identity = self.hparams["lambda_id"] * ((loss_id_A + loss_id_B) / 2)

            # GAN A loss and GAN B loss
            loss_GAN_AB, self.fake_B = self.gan_loss(self.G_AB, self.D_B, real_A, valid)
            loss_GAN_BA, self.fake_A = self.gan_loss(self.G_BA, self.D_A, real_B, valid)
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss: A -> B -> A  and  B -> A -> B
            loss_cycle_A = self.cycle_loss(self.fake_B, self.G_BA, real_A)
            loss_cycle_B = self.cycle_loss(self.fake_A, self.G_AB, real_B)
            loss_cycle = self.hparams["lambda_cyc"] * ((loss_cycle_A + loss_cycle_B) / 2)

            # Total loss
            loss_G = loss_GAN + loss_cycle + loss_identity

            self.log_dict({
                    "total_loss_generators": loss_G,
                    "loss_GAN": loss_GAN,
                    "loss_cycle": loss_cycle,
                    "loss_identity": loss_identity,
                }
            )
            return loss_G

        # The second optimizer is to train the D_A discriminator
        elif optimizer_idx == 1:

            # Real loss
            loss_real = self.discriminator_loss(self.D_A, real_A, valid)

            # Fake loss (on batch of previously generated samples)
            loss_fake = self.discriminator_loss(
                self.D_A, self.fake_A_buffer.push_and_pop(self.fake_A).detach(), fake
            )

            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2
            self.log_dict({
                    "total_D_A": loss_D_A,
                    "loss_D_A_real": loss_real,
                    "loss_D_A_fake": loss_fake,
                }
            )
            return loss_D_A


        # The second optimizer is to train the D_B discriminator
        elif optimizer_idx == 2:

            # Real loss
            loss_real = self.discriminator_loss(self.D_B, real_B, valid)

            # Fake loss (on batch of previously generated samples)
            loss_fake = self.discriminator_loss(
                self.D_B, self.fake_B_buffer.push_and_pop(self.fake_B).detach(), fake
            )

            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            self.log_dict({
                    "total_D_B": loss_D_B,
                    "loss_D_B_real": loss_real,
                    "loss_D_B_fake": loss_fake,
                }
            )
            return loss_D_B

        raise RuntimeError("There is an error in the optimizers configuration!")

    def get_image_examples(
        self, real: torch.Tensor, fake: torch.Tensor
    ) -> Sequence[wandb.Image]:
        """
        Given real and "fake" translated images, produce a nice coupled images to log

        :param real: the real images with shape [batch, channel, w, h]
        :param fake: the fake image with shape [batch, channel, w, h]

        :returns: a sequence of wandb.Image to log and visualize the performance
        """
        example_images = []
        for i in range(real.shape[0]):
            couple = torchvision.utils.make_grid(
                [real[i], fake[i]],
                nrow=2,
                normalize=True,
                scale_each=True,
                pad_value=1,
                padding=4,
            )
            example_images.append(
                wandb.Image(
                    couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB"
                )
            )
        return example_images

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]:
        """ Implements a single validation step

        In each validation step some translation examples are produced and a
        validation loss that uses the cycle consistency is computed
        
        :param batch: the current validation batch
        :param batch_idx: the index of the current validation batch

        :returns: the loss and example images
        """

        real_B = batch["B"]
        fake_A = self.G_BA(real_B)
        images_BA = self.get_image_examples(real_B, fake_A)

        real_A = batch["A"]
        fake_B = self.G_AB(real_A)
        images_AB = self.get_image_examples(real_A, fake_B)

        ####

        real_A = batch["A"]
        real_B = batch["B"]

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        # Cycle loss A -> B -> A
        recov_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(recov_A, real_A)

        # Cycle loss B -> A -> B
        recov_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(recov_B, real_B)

        # Cycle loss aggregation
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
        loss_cycle = self.hparams["lambda_cyc"] * loss_cycle

        # Total loss
        loss_G = loss_cycle

        return {"val_loss": loss_G, "images_BA": images_BA, "images_AB": images_AB}


    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, Union[torch.Tensor,Sequence[wandb.Image]]]]]:
        """ Implements the behaviouir at the end of a validation epoch

        Currently it gathers all the produced examples and log them to wandb,
        limiting the logged examples to `hparams["log_images"]`.

        Then computes the mean of the losses and returns it. 
        Updates the progress bar label with this loss.

        :param outputs: a sequence that aggregates all the outputs of the validation steps

        :returns: the aggregated validation loss and information to update the progress bar
        """
        images_AB = []
        images_BA = []

        for x in outputs:
            images_AB.extend(x["images_AB"])
            images_BA.extend(x["images_BA"])

        images_AB = images_AB[: self.hparams["log_images"]]
        images_BA = images_BA[: self.hparams["log_images"]]

        if not self.is_sanity:  # ignore if it not a real validation epoch. The first one is not.
            print(f"Logged {len(images_AB)} images for each category.")

            self.logger.experiment.log(
                {f"images_AB": images_AB, f"images_BA": images_BA,},
                step=self.global_step,
            )
        self.is_sanity = False

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log_dict({"val_loss": avg_loss})
        return {"val_loss": avg_loss}


def _debug_CycleGANLightningModule(cfg: omegaconf.DictConfig):
    CycleGAN_pl_module = CycleGANLightningModule(
        config=cfg.nn.module,
        optimizer=cfg.nn.model.optimizer,
        input_shape=cfg.nn.model.input_shape,
        num_residual_blocks=cfg.nn.model.num_residual_blocks,
        conv_init=cfg.nn.model.conv_init,
        instancenorm_init=cfg.nn.model.instancenorm_init,
        conv_freeze_parameters=cfg.nn.model.conv_freeze_parameters,
        instancenorm_freeze_parameters=cfg.nn.model.instancenorm_freeze_parameters,
        lr=cfg.nn.model.optimizer.lr,
        beta_1=cfg.nn.model.optimizer.beta_1,
        beta_2=cfg.nn.model.optimizer.beta_2,
        num_epochs=cfg.nn.model.lr_scheduler.num_epochs,
        decay_epoch=cfg.nn.model.lr_scheduler.decay_epoch

    )

    # print(CycleGAN_pl_module)

    out = CycleGAN_pl_module(torch.rand((10, 3, 128, 128)), True)
    # print(out)
    print(out.shape)
    
    out = CycleGAN_pl_module(torch.rand((10, 3, 128, 128)), False)
    # print(out)
    print(out.shape)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    # _debug_AdvancedResNetLighningModule(cfg=cfg)
    _debug_CycleGANLightningModule(cfg=cfg)


if __name__ == "__main__":
    main()
