from math import floor
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchvision
from torchvision import transforms

import pandas as pd
from PIL import Image

from ood.utils.errors import MisconfigurationError


class DatasetWrapper(Dataset):
    """Wrapper class for torch datasets to allow for easy non-iid splitting"""
    def __init__(self):
        self.targets = torch.tensor([])
        self.y_dim = 0

    def __len__(self):
        return len(self.targets)

    @abstractmethod
    def __getitem__(self, i):
        pass

    def get_dims(self):
        """Get the x and y dimensions of the dataset"""
        if len(self) < 1:
            return (0, 0)
        x, _ = self[0]
        return (x.shape[0], self.y_dim)

    def get_idx(self, classes):
        """Get the ids of data belong to the specified classes"""
        return torch.arange(len(self.targets))[
            sum([(self.targets == i).long() for i in classes]).bool()
        ]

    def assign_to_classes(self, classes):
        """Leave only data belonging to the classes within this set"""
        idx = self.get_idx(classes)
        self.data = self.data[idx]
        self.targets = self.targets[idx]


class MNIST(DatasetWrapper):
    """The MNIST dataset in torch readable form"""
    def __init__(self, ds_path, train=True, download=False, classes=None):
        super().__init__()
        ds = torchvision.datasets.MNIST(
            ds_path,
            train=train,
            download=download
        )
        self.data = ds.data.flatten(1).float()
        self.targets = ds.targets
        self.y_dim = 10
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i])


class FashionMNIST(DatasetWrapper):
    """The Fashion MNIST dataset in torch readable form"""
    def __init__(self, ds_path, train=True, download=False, classes=None):
        super().__init__()
        ds = torchvision.datasets.FashionMNIST(
            ds_path,
            train=train,
            download=download
        )
        self.data = ds.data.flatten(1).float()
        self.targets = ds.targets
        self.y_dim = 10
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i])


class KDD99(DatasetWrapper):
    """The KDD Cup99 dataset in torch readable form"""
    def __init__(self, ds_path, train=True, download=False, classes=None):
        super().__init__()
        self.data = torch.tensor([])
        self.targets = torch.tensor([])
        df = pd.read_csv(
            f"{ds_path}/{'train' if train else 'test'}/kddcup.data",
            header=None,
            iterator=True
        )
        nl = 0
        data_len = round(494021 * (0.7 if train else 0.3))
        read_amount = 100_000
        marker = floor(data_len / read_amount) * read_amount
        while read_amount > 0 and (nl := nl + read_amount) <= marker:
            line = df.read(read_amount)
            line = torch.from_numpy(line.to_numpy(np.dtype('float32')))
            data = line[:, 1:-1]
            self.data = torch.cat((self.data, data))
            self.targets = torch.cat((self.targets, line[:, -1]))
            if nl == marker:
                marker = data_len
                read_amount = data_len % read_amount
        self.y_dim = 23
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i].long())


class Amazon(DatasetWrapper):
    """The Amazon dataset in torch readable form"""
    def __init__(self, ds_path, train=True, download=False, classes=None):
        super().__init__()
        df = pd.read_csv(
            f"{ds_path}/{'train' if train else 'test'}/amazon.data",
            header=None
        )
        data = df.to_numpy(np.dtype('float32'))
        self.data = torch.from_numpy(data[:, :-1])
        self.targets = torch.from_numpy(data[:, -1])
        self.y_dim = len(self.targets.unique())
        if classes:
            self.assign_to_classes(classes)

    def __getitem__(self, i):
        return (self.data[i], self.targets[i].long())


class VGGFace(DatasetWrapper):
    """The VGGFace dataset in torch readable form"""
    def __init__(self, ds_path, train=True, download=False, classes=None):
        super().__init__()
        self.ds_path = f"{ds_path}/data"
        self.data_paths = []
        self.targets = []
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.train = train
        composition = [transforms.Resize(256)]
        if train:
            composition += [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            composition += [transforms.CenterCrop(224)]
        composition.append(transforms.ToTensor())
        composition.append(normalize)
        self.transform = transforms.Compose(composition)
        file_info = pd.read_csv(f"{ds_path}/top10_files.csv")
        unique_classes = set()
        for _, r in file_info[file_info['train_flag'] == int(not train)].iterrows():
            if r['Class_ID'] not in unique_classes:
                unique_classes = unique_classes.union({r['Class_ID']})
            if not classes or r['Class_ID'] in classes:
                self.data_paths.append(f"{self.ds_path}/{r['Class_ID']}/{r['file']}")
                self.targets.append(r['Class_ID'])
        self.y_dim = len(unique_classes)
        self.data_paths = np.array(self.data_paths)
        self.targets = torch.tensor(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = Image.open(self.data_paths[idx])
        X = self.transform(X)
        return (X, self.targets[idx].long())


def load_data(ds_name, batch_size, train=True, shuffle=True, classes=None):
    """Load the specified dataset in a form suitable for the model

    Parameters
    ----------
    ds_name: str
        Name of the dataset
    batch_size: int
        Size of batchs of data that may be retrieved
    train: bool
        Load the training dataset if true otherwise load the validation
    shuffle: bool
        Shuffle the dataset it true
    classes: list
        Use only the classes in list, use all classes if empty list or None
    """
    datasets = {
        "mnist": MNIST,
        "fmnist": FashionMNIST,
        "kddcup99": KDD99,
        "amazon": Amazon,
        "vggface": VGGFace,
    }
    if (chosen_set := datasets.get(ds_name)) is None:
        raise MisconfigurationError(
            f"Dataset '{ds_name}' does not exist, " +
            f"possible options: {set(datasets.keys())}"
        )
    data = chosen_set(
        f"./data/{ds_name}",
        train=train,
        download=True,
        classes=classes
    )
    x_dim, y_dim = data.get_dims()
    return {
        "dataloader": torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
        ),
        "x_dim": x_dim,
        "y_dim": y_dim,
    }
