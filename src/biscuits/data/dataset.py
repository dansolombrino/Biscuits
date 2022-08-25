from random import shuffle
from typing import Any
from xmlrpc.client import boolean

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

from pathlib import Path
from typing import Optional, Callable, List, Dict
from PIL import Image
import PIL
import random




from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


def _get_summary_statistics(data_loader):
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in data_loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            
            num_batches += 1

        mean = channels_squared_sum/num_batches
        std = (channels_squared_sum/num_batches - mean ** 2) ** 0.5

        return mean, std


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


class EuroSAT_X_Food_101Dataset(Dataset):
    
    TRAIN_SPLIT_SUMMARY_STATISTICS = dict()
    TRAIN_SPLIT_SUMMARY_STATISTICS["mean"] = torch.tensor(
        [0.2639, 0.2117, 0.1778]
    )
    TRAIN_SPLIT_SUMMARY_STATISTICS["std_dev"] = torch.tensor(
        [0.4408, 0.4085, 0.3824]
    )
    
    def __init__(self, train: bool, path: str, transform, batch_size = 128):
        self.train = train
        self.path = path
        self.transform = transform
        self.batch_size = batch_size

        self.image_datasets = datasets.ImageFolder(
            self.path, self.transform
        )
        
        # This is how summary stats have been computed for the dataset
        # self.dataloaders = torch.utils.data.DataLoader(
        #     self.image_datasets, 
        #     batch_size=self.batch_size, 
        #     shuffle=True, 
        #     num_workers=16
        # )
        # summary_stats = _get_summary_statistics(self.dataloaders)
        # print(f"mean   : {summary_stats[0]}")
        # print(f"std_dev: {summary_stats[1]}")
        
        self.class_names = self.image_datasets.classes


    def __len__(self):
        return len(self.image_datasets)

    def __getitem__(self, index):
        return self.image_datasets[index]


    def __repr__(self) -> str:
        return f"--- EuroSAT_X_Food_101Dataset ---\n\n" + \
            f"len: {self.__len__()}\n" + \
            f"classes: {self.class_names}\n" + \
            f"train: {self.train}\n" + \
            f"path : {self.path}\n" + \
            f"transform: {self.transform}" + \
            "\n\n---------------------------"


def _debug_EuroSAT_X_Food_101Dataset(is_training, path, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            EuroSAT_X_Food_101Dataset.TRAIN_SPLIT_SUMMARY_STATISTICS["mean"], 
            EuroSAT_X_Food_101Dataset.TRAIN_SPLIT_SUMMARY_STATISTICS["std_dev"],
        )
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(
            EuroSAT_X_Food_101Dataset.TRAIN_SPLIT_SUMMARY_STATISTICS["mean"], 
            EuroSAT_X_Food_101Dataset.TRAIN_SPLIT_SUMMARY_STATISTICS["std_dev"],
        )
    ])

    eurosat_x_food_101_dataset = EuroSAT_X_Food_101Dataset(
        train=is_training, 
        path=path,
        transform=train_transform if is_training else test_transform,
        batch_size=batch_size
    )

    print(eurosat_x_food_101_dataset)


class EuroSATDataset(Dataset):

    TRAIN_SPLIT_SUMMARY_STATISTICS = dict()
    
    TRAIN_SPLIT_SUMMARY_STATISTICS["1200"] = dict()
    TRAIN_SPLIT_SUMMARY_STATISTICS["1200"]["mean"] = torch.tensor(
        [0.1567, 0.1627, 0.1815]
    )
    TRAIN_SPLIT_SUMMARY_STATISTICS["1200"]["std_dev"] = torch.tensor(
        [0.3636, 0.3691, 0.3854]
    )
    
    TRAIN_SPLIT_SUMMARY_STATISTICS["30000"] = dict()
    TRAIN_SPLIT_SUMMARY_STATISTICS["30000"]["mean"] = torch.tensor(
        [0.1604, 0.1636, 0.1798]
    )
    TRAIN_SPLIT_SUMMARY_STATISTICS["30000"]["std_dev"] = torch.tensor(
        [0.3669, 0.3699, 0.3840]
    )
    
    def __init__(self, train: bool, path: str, transform, batch_size = 128):
        self.train = train
        self.path = path
        self.transform = transform
        self.batch_size = batch_size

        self.image_datasets = datasets.ImageFolder(
            self.path, self.transform
        )
        
        # This is how summary stats have been computed for the dataset
        # self.dataloaders = torch.utils.data.DataLoader(
        #     self.image_datasets, 
        #     batch_size=self.batch_size, 
        #     shuffle=True, 
        #     num_workers=16
        # )
        # summary_stats = _get_summary_statistics(self.dataloaders)
        # print(f"mean   : {summary_stats[0]}")
        # print(f"std_dev: {summary_stats[1]}")
        
        self.class_names = self.image_datasets.classes


    def __len__(self):
        return len(self.image_datasets)

    def __getitem__(self, index):
        return self.image_datasets[index]


    def __repr__(self) -> str:
        return f"--- EuroSAT_X{self.__len__()}Dataset ---\n\n" + \
            f"len: {self.__len__()}\n" + \
            f"classes: {self.class_names}\n" + \
            f"train: {self.train}\n" + \
            f"path : {self.path}\n" + \
            f"transform: {self.transform}" + \
            "\n\n---------------------------"


def _debug_EuroSATDataset(is_training, path, batch_size):
    num_samples = path.split("_")[1]
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            EuroSATDataset.TRAIN_SPLIT_SUMMARY_STATISTICS[
                str(num_samples)
            ]["mean"], 
            EuroSATDataset.TRAIN_SPLIT_SUMMARY_STATISTICS[
                str(num_samples)
            ]["std_dev"],
        )
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        transforms.Normalize(
            EuroSATDataset.TRAIN_SPLIT_SUMMARY_STATISTICS[
                str(num_samples)
            ]["mean"], 
            EuroSATDataset.TRAIN_SPLIT_SUMMARY_STATISTICS[
                str(num_samples)
            ]["std_dev"],
        )
    ])

    eurosat_dataset = EuroSATDataset(
        train=is_training, 
        path=path,
        transform=train_transform if is_training else test_transform,
        batch_size=batch_size
    )

    print(eurosat_dataset)


class DatasetUnpaired(Dataset):

    def __init__(self, 
                 folderA: Path, 
                 folderB: Path, 
                 trainA: bool,
                 trainB: bool,
                 transform: Optional[Callable] = None,
                 fixed_pairs: bool = False,
        ) -> None:
        """
        Dataset to handle unpaired images, i.e. the number of images in folderA
        and in folderB may be different.

        :param folderA: path to the folder that contains the A images
        :param folderB: path to the folder that contains the B images
        :param tranform: tranform to apply to the images
        """
        super().__init__()
        self.folderA: Path = Path(folderA)
        self.folderB: Path = Path(folderB)

        self.trainA, self.trainB = trainA, trainB

        if not (folderA.is_dir() and folderB.is_dir()):
            raise RuntimeError(f"The folders are not valid!\n\t- Folder A: {folderA}\n\t- Folder B: {folderB}")

        self.filesA: List[Path] = list(sorted(folderA.rglob('*.jpg')))
        self.filesB: List[Path] = list(sorted(folderB.rglob('*.jpg')))

        if not self.filesA:
            raise RuntimeError("Empty image lists for folderA!")
        
        if not self.filesB:
            raise RuntimeError("Empty image lists for folderB!")

        self.filesA_num: int = len(self.filesA)
        self.filesB_num: int = len(self.filesB)
        
        self.transform: Optional[Callable] = transform
        self.fixed_pairs: bool = fixed_pairs

    def __len__(self) -> int:
        """
        Since it is unpaired, it is not well defined.
        We will use the maximum number of images between folderA and folderB

        :returns: maximum number between #imagesA and #imagesB
        """
        return max(self.filesA_num, self.filesB_num)

    def pil_loader(self, path: Path) -> PIL.Image:
        """ PIL loader implementation from the Pytorch's ImageDataset class
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder

        :param path: the path to an image
        :returns: an image converted into RGB format
        """
        # open path as file to avoid ResourceWarning 
        # (https://github.com/python-pillow/Pillow/issues/835)

        # print(f"[PIL Loader] opening file: {str(path)}")
        with path.open('rb') as f:
            img = PIL.Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Return a random sample imageA-imageB

        :param index: index of the sample (not relevant)
        :returns: a dictionary containing:
                    - A: the imageA
                    - B: the imageB
                    - pathA: the path to the imageA
                    - pathB: the path to the imageB
        """

        # Enforce a valid index for `filesA`
        fileA = self.filesA[index % self.filesA_num]

        if self.fixed_pairs:
            # When e.g. testing use reproducible samples
            fileB = self.filesB[index % self.filesB_num]

        else:
            # When training, get a random image from filesB
            fileB = self.filesB[random.randint(0, self.filesB_num - 1)]

        imageA = self.pil_loader(fileA)

        imageB = self.pil_loader(fileB)

        if self.transform is not None:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)
        
        return {
            'A': imageA,
            'A_path': str(fileA),
            'B': imageB,
            'B_path': str(fileB),
        }


def _debug_DatasetUnpaired(
        folderA: Path, 
        folderB: Path, 
        trainA: bool,
        trainB: bool,
        fixed_pairs: bool,
        split: str,
        img_height: int,
        img_width: int
):

    if trainA != trainB:
        raise(
            ValueError(
                f"trainA and trainB should be the same, got instead: " + 
                f"trainA --> {trainA}, trainB --> {trainB}"
            )
        )

    if split == "train":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    int(img_height * 1.12), Image.BICUBIC
                ),
                transforms.RandomCrop(
                    (img_height, img_width)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    elif split == "validation":
        transform = transforms.Compose(
            [
                transforms.Resize(
                    int(img_height * 1.12), Image.BICUBIC
                ),
                transforms.CenterCrop(
                    (img_height, img_width)
                ),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = None

    dataset_unpaired = DatasetUnpaired(
        folderA=folderA,
        folderB=folderB,
        trainA=trainA,
        trainB=trainB,
        transform=transform,
        fixed_pairs=fixed_pairs
    )

    print(dataset_unpaired.__getitem__(50))

    ### TAKES HELLA TIME ###
    # for i in tqdm(dataset_unpaired):
    #     pass


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    # _debug_AntsVsBeesDataset(
    #     cfg.data.datasets.train_set.train, cfg.data.datasets.train_set.path
    # )

    # _debug_EuroSATDataset(
    #     is_training=cfg.data.datasets.train_set.train, 
    #     path=cfg.data.datasets.train_set.path,
    #     batch_size=cfg.data.batch_size.train
    # )

    _debug_DatasetUnpaired(
        folderA = Path(cfg.data.datasets.test_set[0].folderA), 
        folderB = Path(cfg.data.datasets.test_set[0].folderB), 
        trainA=cfg.data.datasets.test_set[0].trainA,
        trainB=cfg.data.datasets.test_set[0].trainB,
        fixed_pairs=cfg.data.datasets.fixed_pairs,
        split="test",
        img_height=cfg.data.img_height,
        img_width=cfg.data.img_width
    )



if __name__ == "__main__":
    main()    
