from typing import Any

import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageNet
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        train: bool,
        path: str,
        split: str,
        transform: Any
        # **kwargs
    ):
        super().__init__()
        self.train = train
        self.path = path
        self.split = split
        self.transform = transform

        self.cifar10 = CIFAR10(
            root=self.path,
            train=self.train,
            download=True,
            transform=self.transform,
        )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, index: int):
        return self.cifar10[index]

    def __repr__(self) -> str:
        return f"CIFAR10Dataset(n_instances={len(self)})"


class ImageNetDataset(Dataset):
    def __init__(
        self,
        train: bool,
        path: str,
        # **kwargs
    ):
        super().__init__()
        self.train = train
        self.path = path

        self.imagenet = ImageNet(
            root=self.path, train=self.train, download=True
        )

    def __len__(self) -> int:
        return len(self.imagenet)

    def __getitem__(self, index: int):
        return self.imagenet[index]

    def __repr__(self) -> str:
        return f"ImageNetDataset(n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    dataset: CIFAR10Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train_set, _recursive_=False
    )

    print(dataset[0])


if __name__ == "__main__":
    main()
