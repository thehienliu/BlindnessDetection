import os
import pandas as pd
from PIL import Image
from typing import Dict
import opendatasets as od
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class APTOS2019Dataset(Dataset):
    def __init__(
        self,
        root: str = ".",
        split_ratio: Dict[str, int] = {"train": 0.65, "val": 0.15, "test": 0.2},
        download: bool = True,
        dataset_split: str = "train",
        transform=None,
        default_size=256,
    ):

        # Download dataset
        exist_dir = os.path.isdir(f"{root}/aptos2019-blindness-detection")
        if download and not exist_dir:
            APTOS2019Dataset.download(root)
        elif not download and not exist_dir:
            raise ValueError(
                "No such file for prepare dataset, please download the dataset first!"
            )

        # setup
        data_csv = pd.read_csv(f"{root}/aptos2019-blindness-detection/train.csv")
        self.image_dir = f"{root}/aptos2019-blindness-detection/train_images"
        self.dataset_split = dataset_split
        self.default_size = default_size
        self.transform = self.set_transform(transform)
        self.class_names = {
            "No-DR": 0,
            "Mild": 1,
            "Moderate": 2,
            "Severe": 3,
            "Proliferative-DR": 4,
        }
        self.data = self.split_dataset(data_csv, split_ratio)

    @staticmethod
    def download(root: str = ".", force: bool = False) -> None:
        od.download(
            "https://www.kaggle.com/competitions/aptos2019-blindness-detection",
            data_dir=root,
            force=force,
        )

    def split_dataset(
        self,
        data_csv: pd.DataFrame,
        split_ratio: Dict[str, int],
        shuffle: bool = True,
        random_state: int = 42,
    ):

        train_val_data, test_data = train_test_split(
            data_csv,
            test_size=split_ratio["test"],
            random_state=random_state,
            shuffle=True,
            stratify=data_csv.diagnosis.values,
        )

        train_data, val_data = train_test_split(
            train_val_data,
            test_size=split_ratio["val"],
            random_state=random_state,
            shuffle=True,
            stratify=train_val_data.diagnosis.values,
        )
        if self.dataset_split == "train":
            data = train_data.values
        elif self.dataset_split == "val":
            data = val_data.values
        else:
            data = test_data.values

        return data

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        if transform:
            return transform

        return transforms.Compose(
            [
                transforms.Resize((self.default_size, self.default_size)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        image_id, label = self.data[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.png")
        image = Image.open(image_path)

        return {"image_id": image_id, "image": self.transform(image), "label": label}

    def __repr__(self) -> str:
        body = f"""Dataset {self.__class__.__name__}
    Set: {self.dataset_split}
    Number of datapoints: {len(self.data)}
    Image location: {self.image_dir}
    Transform: {self.transform}"""
        return body
