import logging
from pickletools import optimize
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from biscuits.data.datamodule import MetaData
from biscuits.modules import Advanced_ResNet, Basic_ResNet
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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _debug_AdvancedResNetLighningModule(cfg=cfg)


if __name__ == "__main__":
    main()
