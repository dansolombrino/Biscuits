import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

from pprint import pprint

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        # example
        # self.class_vocab: Mapping[str, int] = class_vocab

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        # pylogger.debug(f"Saving MetaData to '{dst_path}'")

        # example
        # (dst_path / "class_vocab.tsv").write_text(
        #     "\n".join(f"{key}\t{value}" for key, value in self.class_vocab.items())
        # )

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        # example
        # lines = (
        #     (src_path / "class_vocab.tsv")
        #     .read_text(encoding="utf-8")
        #     .splitlines()
        # )

        # class_vocab = {}
        # for line in lines:
        #     key, value = line.strip().split("\t")
        #     class_vocab[key] = value

        # return MetaData(
        #     class_vocab=class_vocab,
        # )
        return MetaData()


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return default_collate(samples)


# class MyDataModule(pl.LightningDataModule):
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        validation_percentage_split: float,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"

        self.train_dataset: Optional[Dataset] = None
        self.validation_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.validation_percentage_split: float = validation_percentage_split

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        # return MetaData(class_vocab=self.train_dataset.dataset.class_vocab)
        return MetaData()

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=(0, 0),
                    translate=(0.0625, 0.0625)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), 
                    (0.247, 0.243, 0.261)
                )
                # transforms.ToTensor()
            ]
        )

        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (
            self.train_dataset is None and self.validation_datasets is None
        ):

            train_set = hydra.utils.instantiate(
                config=self.datasets.train_set,
                train=True,
                path=self.datasets.train_set.path,
                transform=transform, # TODO pass it via hydra
            )
            
            train_set_length = int(
                len(train_set) * (1 - self.validation_percentage_split)
            )

            validation_set_length = len(train_set) - train_set_length

            self.train_dataset, validation_dataset = random_split(
                train_set, [train_set_length, validation_set_length]
            )

            self.validation_datasets = [validation_dataset]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    config=test_set_cfg,
                    train=False,
                    # path=self.datasets.test_set.path,
                    path=test_set_cfg.path,
                    transform=transform,
                ) for test_set_cfg in self.datasets.test_set
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                collate_fn, split="train", metadata=self.metadata
            ),
        )
    
    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=partial(
                    collate_fn, split="val", metadata=self.metadata
                ),
            )
            for dataset in self.validation_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                collate_fn=partial(
                    collate_fn, split="test", metadata=self.metadata
                ),
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


class AntsVsBeesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        validation_percentage_split: float,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"

        self.train_dataset: Optional[Dataset] = None
        self.validation_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.validation_percentage_split: float = validation_percentage_split

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        # return MetaData(class_vocab=self.train_dataset.dataset.class_vocab)
        return MetaData()

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])        

        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (
            self.train_dataset is None and self.validation_datasets is None
        ):

            self.train_dataset = hydra.utils.instantiate(
                config=self.datasets.train_set,
                train=True,
                path=self.datasets.train_set.path,
                transform=train_transform
            )

        if stage is None or stage == "test":
            # see CIFAR10 for support of union of multiple dataset!

            test_set = hydra.utils.instantiate(
                config=self.datasets.test_set,
                train=False,
                # path=self.datasets.test_set.path,
                path=self.datasets.test_set.path,
                transform=test_transform,
            )

            val_set_len = len(test_set) * self.validation_percentage_split
            test_set_len = len(test_set) - val_set_len

            self.validation_datasets, self.test_datasets = random_split(
                test_set, [val_set_len, test_set_len]
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(
                collate_fn, split="train", metadata=self.metadata
            ),
        )
    
    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=partial(
                    collate_fn, split="val", metadata=self.metadata
                ),
            )
            for dataset in self.validation_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
                collate_fn=partial(
                    collate_fn, split="test", metadata=self.metadata
                ),
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config=cfg.data,
        _recursive_=False,
    )

    # print(datamodule)


if __name__ == "__main__":
    main()
