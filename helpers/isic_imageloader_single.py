import random
import logging
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader, SubsetRandomSampler

from helpers.isic_utilities import create_datasets, create_dataframe

log = logging.getLogger(__name__)

class IsicImagesLoader(pl.LightningDataModule):
    """
    Class to pull the zip file from google drive and returns the data loaders
    for the training, validation and tests sets. It receives as input the path of
    the zip file, the percentage of the training set that is going to be used for validation
    and the transformations to be applied to the train/val sets and the test set.
    """

    def __init__(self,
                 images_path=None,
                 src_path=None,
                 val_percentage=0.2,
                 num_samples = 6000,
                 train_batch_size=1,
                 train_transformations=[],
                 test_transformations=[],
                 seed=None):
        super().__init__()
        self.num_workers = 0
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
            pl.seed_everything(seed)

        if not images_path or not src_path:
            raise Exception('Images path and source path are required')
        else:
            self.image_path = images_path
            self.source_path = src_path

        self.num_samples = num_samples
        self.val_percentage = val_percentage
        self.train_batch_size = train_batch_size
        self.train_transformations = train_transformations
        self.test_transformations = test_transformations

        self.train_dataset = None
        self.test_dataset = None

    def get_class_indices(self):
        """
        Returns the classes found.
        """
        return self.idx2class


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

    def setup(self, stage=None):
        """
        Executed automatically by lightning when a dataloader is requested.
        """
        self.get_datasets()
        self._train_val_samplers()

    def get_datasets(self):
        df = create_dataframe(self.image_path, self.source_path, self.num_samples)
        self.train_dataset, self.test_dataset = create_datasets(df, test_size=0.2,
                                                                train_transform=self.train_transformations,
                                                                test_transform=self.test_transformations)
        idx2class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.idx2class = idx2class

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
        return DataLoader(dataset=self.test_dataset, shuffle=False, batch_size=1, num_workers=self.num_workers)