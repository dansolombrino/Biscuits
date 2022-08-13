import logging
from typing import List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import biscuits  # noqa
from biscuits.data.datamodule import MetaData

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(
            f"Adding callback <{callback['_target_'].split('.')[-1]}>"
        )
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(
            f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.data['_target_']}>")

    if "CIFAR10" in cfg.data['_target_']:
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(
            # config=cfg.nn.data,
            # datasets=cfg.nn.data.datasets, 
            # num_workers=cfg.nn.data.num_workers,
            # batch_size=cfg.nn.data.batch_size,
            # validation_percentage_split=cfg.nn.data.validation_percentage_split,
            # _recursive_=False,
            config=cfg.data,
            datasets=cfg.data.datasets, 
            num_workers=cfg.data.num_workers,
            batch_size=cfg.data.batch_size,
            validation_percentage_split=cfg.data.validation_percentage_split,
            _recursive_=False,
        )
    elif "AntsVsBees" in cfg.data["_target_"]:
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(
            config=cfg.data,
            datasets=cfg.data.datasets, 
            num_workers=cfg.data.num_workers,
            batch_size=cfg.data.batch_size,
            _recursive_=False,
        )
    elif "EuroSAT_X_Food_101" in cfg.data["_target_"]:
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(
            config=cfg.data,
            datasets=cfg.data.datasets, 
            num_workers=cfg.data.num_workers,
            batch_size=cfg.data.batch_size,
            _recursive_=False,
        )
    else:
        raise KeyError(
            f"{cfg.data['_target_']} DataModule does NOT exist"
        )

    metadata: Optional[MetaData] = getattr(datamodule, "metadata", None)
    if metadata is None:
        pylogger.warning(
            f"No 'metadata' attribute found in datamodule <{datamodule.__class__.__name__}>"
        )

    # TODO 
    # to give the ability of picking model from Hydra conf, use a switch
    # case here.
    # Switch case on a param in Hydra, which stores the name of the model you'd
    # like to use
    # for example, place a str ID of the model in cfg.nn.model_name
    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")

    if "BasicResNetLightningModule" in cfg.nn.module['_target_']:

        model: pl.LightningModule = hydra.utils.instantiate(
            config=cfg.nn.module,
            _recursive_=False,
            metadata=metadata,
            resnet_depth=cfg.nn.model.resnet_depth,
            conv_init_method=cfg.nn.model.conv_init_method,
            batchnorm_init_methods=cfg.nn.model.batchnorm_init_methods,
            lin_init_method=cfg.nn.model.lin_init_method,
            conv_freeze_parameters=cfg.nn.model.conv_freeze_parameters,
            batchnorm_freeze_parameters=cfg.nn.model.batchnorm_freeze_parameters,
            lin_freeze_parameters=cfg.nn.model.lin_freeze_parameters,
            dropout_probability=cfg.nn.model.dropout_probability,
            dropout2d_probability=cfg.nn.model.dropout2d_probability,
            optimizer=cfg.nn.model.optimizer
        )
    
    elif "AdvancedResNetLightningModule" in cfg.nn.module['_target_']:
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
        num_classes=cfg.data.datasets.num_classes,
        _recursive_=False,
    )
    
    # default case
    else: 
        raise KeyError(
            f"No LighningModule named {cfg.nn.module['_target_']}"
        )
        


    # Instantiate the callbacks
    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    callbacks: List[Callback] = build_callbacks(
        cfg.train.callbacks, template_core
    )

    storage_dir: str = cfg.core.storage_dir

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging,
        cfg=cfg,
        resume_id=template_core.resume_id,
    )

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=template_core.trainer_ckpt_path,
    )

    if fast_dev_run:
        pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    else:
        if (
            # "test_set" in cfg.nn.data.datasets
            "test_set" in cfg.data.datasets
            and trainer.checkpoint_callback.best_model_path is not None
        ):
            pylogger.info("Starting testing!")
            trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
