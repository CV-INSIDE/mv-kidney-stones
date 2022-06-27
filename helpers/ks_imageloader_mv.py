import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, SequentialSampler, Sampler
from typing import Sized, Iterator
from pytorch_lightning.trainer.supporters import CombinedLoader
import numpy as np
import os
from torchvision import transforms, utils, datasets
import random


class TestSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(self.data_source)

    def __len__(self) -> int:
        return len(self.data_source)


class KidneyImagesLoader(pl.LightningDataModule):
    """
    Class to pull the zip file from google drive and returns the data loaders
    for the training, validation and tests sets. It receives as input the path of
    the zip file, the percentage of the training set that is going to be used for validation
    and the transformations to be applied to the train/val sets and the test set.
    """

    def __init__(self,
                 zip_path=None,
                 images_path=None,
                 val_percentage=0.2,
                 train_batch_size=1,
                 train_transformations=None,
                 test_transformations=None,
                 seed=None):
        super().__init__()
        self.num_workers = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            pl.seed_everything(seed)
            # torch.manual_seed(seed)
            # if you are suing GPU
            # torch.cuda.manual_seed(seed)
            # torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.enabled = False
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True
            # self.num_workers = 0
        self.train_dataset_1 = None
        self.train_dataset_2 = None
        self.val_sampler_1 = None
        self.val_sampler_2 = None
        self.train_dataset_1 = None
        self.train_dataset_2 = None

        if not zip_path and not images_path:
            raise Exception('Zip path or images path is required')

        if zip_path:
            self.zip_path = zip_path
            self.image_path = self.zip_path[self.zip_path.rindex("/"):]
            self.image_path = "/content" + self.image_path.replace(".zip", "")
        else:
            # Elias
            if len(images_path) == 1:
                self.image_path = images_path
            else:
                self.image_path_view_1 = images_path[0]
                self.image_path_view_2 = images_path[1]

        self.val_percentage = val_percentage
        self.train_batch_size = train_batch_size
        self.train_transformations = train_transformations
        self.test_transformations = test_transformations

    """
    Returns the classes found.
    """

    def get_class_indices(self):
        return self.idx2class

    def get_class_indiices_view1(self):
        return self.idx2class_1

    def get_class_indiices_view2(self):
        return self.idx2class_2

    """
    Returns the training dataset and the mapping for index to class name.
    """

    def get_train_datasets(self):
        dataset_1 = datasets.ImageFolder(root=self.image_path_view_1 + "/train", transform=self.train_transformations)
        dataset_2 = datasets.ImageFolder(root=self.image_path_view_2 + "/train", transform=self.train_transformations)
        idx2class_1 = {v: k for k, v in dataset_1.class_to_idx.items()}
        idx2class_2 = {v: k for k, v in dataset_2.class_to_idx.items()}
        self.idx2class_1 = idx2class_1
        self.train_dataset_1 = dataset_1
        self.idx2class_2 = idx2class_2
        self.train_dataset_2 = dataset_2
        return dataset_1, dataset_2, idx2class_1, idx2class_2

    def _train_val_samplers(self):
        """
      Splits the training set into train and validation sets and it returns the
      generated SubsetRandomSamplers.
      :return:
      """

        # Get indexes and shuffle
        rps_dataset_size = len(self.train_dataset_1)
        rps_dataset_size2 = len(self.train_dataset_2)
        rps_dataset_indices = list(range(rps_dataset_size))
        np.random.shuffle(rps_dataset_indices)

        val_split_index = int(np.floor(self.val_percentage * rps_dataset_size))
        train_idx, val_idx = rps_dataset_indices[val_split_index:], rps_dataset_indices[:val_split_index]

        self.train_sampler_1 = TestSampler(train_idx)
        # self.val_sampler_1 = SubsetRandomSampler(val_idx)

        # self.train_sampler_1 = SequentialSampler(train_idx)
        self.val_sampler_1 = TestSampler(val_idx)

        # View 2
        # ToDo: Implement a method to not randomize second view and make it dependant of the distribution of first view
        # rps_dataset_indices = list(range(rps_dataset_size))
        # np.random.shuffle(rps_dataset_indices)

        # val_split_index = int(np.floor(self.val_percentage * rps_dataset_size))
        # train_idx, val_idx = rps_dataset_indices[val_split_index:], rps_dataset_indices[:val_split_index]

        # self.train_sampler_2 = SubsetRandomSampler(train_idx)
        # self.val_sampler_2 = SubsetRandomSampler(val_idx)

        self.train_sampler_2 = TestSampler(train_idx)
        self.val_sampler_2 = TestSampler(val_idx)

    """
    Returns the test dataset.
    """

    def get_test_datasets(self):
        dataset_test_1 = datasets.ImageFolder(root=self.image_path_view_1 + "/test",
                                              transform=self.test_transformations)
        dataset_test_2 = datasets.ImageFolder(root=self.image_path_view_2 + "/test",
                                              transform=self.test_transformations)
        return dataset_test_1, dataset_test_2

    """
    Executed automatically by lightning when a dataloader is requested.
    """

    def setup(self, stage=None):
        if hasattr(self, 'zip_path'):
            cmd = 'yes | cp "' + self.zip_path + '" -d "/content/"'
            os.system(cmd)
            cmd = 'unzip -o --qq ' + self.image_path + ".zip" + ' -d "' + self.image_path + '"'
            os.system(cmd)
        self.get_train_datasets()
        self._train_val_samplers()

    """
    Returns the train data loader.
    """

    def train_dataloader(self):
        view1_loader = DataLoader(dataset=self.train_dataset_1, shuffle=False, batch_size=self.train_batch_size,
                                  sampler=self.train_sampler_1, num_workers=self.num_workers)
        view2_loader = DataLoader(dataset=self.train_dataset_2, shuffle=False, batch_size=self.train_batch_size,
                                  sampler=self.train_sampler_2, num_workers=self.num_workers)
        return CombinedLoader({"view1": view1_loader, "view2": view2_loader}, 'min_size')

    """
    Returns the validation data loader.
    """

    def val_dataloader(self):
        view1_loader = DataLoader(dataset=self.train_dataset_1, shuffle=False, batch_size=1, sampler=self.val_sampler_1,
                                  num_workers=self.num_workers)
        view2_loader = DataLoader(dataset=self.train_dataset_2, shuffle=False, batch_size=1, sampler=self.val_sampler_2,
                                  num_workers=self.num_workers)
        loader = {"view1": view1_loader, "view2": view2_loader}
        combined = CombinedLoader(loader, 'min_size')
        return combined

    """
    Returns the test data loader.
    """

    def test_dataloader(self):
        dataset1, dataset2 = self.get_test_datasets()
        dataloader1 = DataLoader(dataset=dataset1, shuffle=False, batch_size=1, num_workers=self.num_workers)
        dataloader2 = DataLoader(dataset=dataset2, shuffle=False, batch_size=1, num_workers=self.num_workers)
        loader = {"view1": dataloader1, "view2": dataloader2}
        combined = CombinedLoader(loader, 'min_size')
        return combined
