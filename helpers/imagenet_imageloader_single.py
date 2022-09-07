import os
import random
import logging
import numpy as np
import pytorch_lightning as pl

from torchvision import datasets
from torch.utils.data import DataLoader, SubsetRandomSampler

log = logging.getLogger(__name__)

class ImagenetImagesLoader(pl.LightningDataModule):
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
                 train_transformations=[],
                 test_transformations=[],
                 seed=None,
                 color_transform=False,
                 hsv=False,
                 lbp=False):
        super().__init__()
        self.num_workers = 0
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
            pl.seed_everything(seed)

        if not zip_path and not images_path:
            raise Exception('Zip path or images path is required')

        if zip_path:
            self.zip_path = zip_path
            self.image_path = self.zip_path[self.zip_path.rindex("/"):]
            self.image_path = "/content" + self.image_path.replace(".zip", "")
        else:
            self.image_path = images_path

        self.val_percentage = val_percentage
        self.train_batch_size = train_batch_size
        self.train_transformations = train_transformations
        self.test_transformations = test_transformations

        # Custom, augment data by adding hsv and lbp image transformations
        self.color_transform = color_transform
        self.hsv = hsv
        self.lbp = lbp

    def get_class_indices(self):
        """
        Returns the classes found.
        """
        return self.idx2class

    def get_train_dataset(self):
        """
        Returns the training dataset and the mapping for index to class name.
        """
        dataset = datasets.CIFAR10(root=self.image_path, train=True, download=True, transform=self.train_transformations)

        log.info(f"length of dataset {len(dataset)}")
        idx2class = {v: k for k, v in dataset.class_to_idx.items()}
        self.idx2class = idx2class
        self.train_dataset = dataset
        return dataset, idx2class

    def _train_val_samplers(self):
        """
        Splits the training set into train and validation sets and it returns the
        generated SubsetRandomSamplers.
        """
        rps_dataset_size = len(self.train_dataset)
        rps_dataset_indices = list(range(rps_dataset_size))
        np.random.shuffle(rps_dataset_indices)

        val_split_index = int(np.floor(self.val_percentage * rps_dataset_size))
        train_idx, val_idx = rps_dataset_indices[val_split_index:], rps_dataset_indices[:val_split_index]

        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)

        log.info(f"train batch size: {train_idx}, val batch size: {val_idx}")

    def get_test_dataset(self):
        """
        Returns the test dataset.
        """
        dataset_test = datasets.CIFAR10(root=self.image_path, train=False, download=True,
                                        transform=self.test_transformations)
        return dataset_test

    def setup(self, stage=None):
        """
        Executed automatically by lightning when a dataloader is requested.
        """
        if hasattr(self, 'zip_path'):
            cmd = 'yes | cp "' + self.zip_path + '" -d "/content/"'
            os.system(cmd)
            cmd = 'unzip -o --qq ' + self.image_path + ".zip" + ' -d "' + self.image_path + '"'
            os.system(cmd)
        self.get_train_dataset()
        self._train_val_samplers()

    def train_dataloader(self):
        """
        Returns the train data loader.
        """
        return DataLoader(dataset=self.train_dataset, shuffle=False, batch_size=self.train_batch_size,
                          sampler=self.train_sampler, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns the validation data loader.
        """
        return DataLoader(dataset=self.train_dataset, shuffle=False, batch_size=1, sampler=self.val_sampler,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns the test data loader.
        """
        dataset = self.get_test_dataset()
        log.info(f"length of test dataset {len(dataset)}")
        return DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=self.num_workers)