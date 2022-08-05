from random import shuffle
from typing import Any

import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageNet
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image


from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        train: bool,
        path: str,
        split: str,
        transform: Any,
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
            # turn this off, while debugging, and turn on debug return in 
            # __getitem__ method
            transform=self.transform,
        )

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, index: int):
        # use this in production
        return self.cifar10[index]
        
        # use this only when debugging, because it allows to see data before and
        # after transformations
        # return (
        #     self.transform(self.cifar10[index][0]),
        #     self.cifar10[index][1],
        #     self.cifar10[index][0],
        # )

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

    dataset: CIFAR10Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train_set,
        split="train",
        transform=transform,
        # path=PROJECT_ROOT / "data",
        path=cfg.nn.data.datasets.train_set.path,
    )

    for i in range(0, 10):
        save_image(dataset[i][0], "/tmp/augmented_imgs/" + str(i) + "_aug.png")
        save_image(
            transforms.ToTensor()(dataset[i][2]), 
            "/tmp/augmented_imgs/" + str(i) + "_og.png"
        )

    # print(type(dataset[0][1]))


if __name__ == "__main__":
    main()
