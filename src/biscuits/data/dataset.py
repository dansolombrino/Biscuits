from random import shuffle
from typing import Any

import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageNet
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import datasets

import os
from pprint import pprint


from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        train: bool,
        path: str,
        transform: Any,
        # **kwargs
    ):
        super().__init__()
        self.train = train
        self.path = path
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


def _debug_CIFAR10Dataset(): 

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

    print(type(dataset[0][1]))


class AntsVsBeesDataset(Dataset):
    def __init__(self, train: bool, path: str, transform):
        self.train = train
        self.path = path
        self.transform = transform

        self.image_datasets = datasets.ImageFolder(
            self.path, self.transform
        )
        
        self.dataloaders = torch.utils.data.DataLoader(
            self.image_datasets, batch_size=4, shuffle=True, num_workers=4
        )
        
        self.class_names = self.image_datasets.classes

    def __len__(self):
        return len(self.image_datasets)

    def __getitem__(self, index):
        return self.image_datasets[index]

    def __repr__(self) -> str:
        return f"--- AntsVsBeesDataset ---\n\n" + \
            f"len: {self.__len__()}\n" + \
            f"classes: {self.class_names}\n" + \
            f"train: {self.train}\n" + \
            f"path : {self.path}\n" + \
            f"transform: {self.transform}" + \
            "\n\n---------------------------"


def _debug_AntsVsBeesDataset(is_training, path):
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

    ants_vs_bees_dataset = AntsVsBeesDataset(
        train=is_training, 
        path=path,
        transform=train_transform if is_training else test_transform
    )

    print(ants_vs_bees_dataset)



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _debug_AntsVsBeesDataset(
        cfg.data.datasets.train_set.train, cfg.data.datasets.train_set.path
    )


if __name__ == "__main__":
    main()    
