import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, FashionMNIST

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split


class MyDataset(Dataset):
    def __init__(
        self,
        train: bool,
        path: str,
        # **kwargs
    ):
        super().__init__()
        self.train = train
        self.path = path

        self.cifar10 = CIFAR10(root=self.path, train=self.train, download=True)

    def __len__(self) -> int:
        return len(self.cifar10)

    def __getitem__(self, index: int):
        return self.cifar10[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    dataset: MyDataset = hydra.utils.instantiate(cfg.nn.data.datasets.train, _recursive_=False)


if __name__ == "__main__":
    main()
