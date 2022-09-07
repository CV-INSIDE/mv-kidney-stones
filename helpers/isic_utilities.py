"""
File used to load ISIC dataset
"""
import os
import torch
import typing
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

import helpers.data_loader as data_loader


def create_dataframe(image_path, csv_file, num_samples: typing.Union[str, int] = 'all'):
    """
    Creates a dataframe from the csv_file that consists of two rows. The first row will contain the concatenation of
    the base image_path and the image file name, and the second column will contain the label type.
    Args:
        image_path (str): Absolute path to image folder
        csv_file (str): Absolute path to the csv file containing ISIC dataset files
        num_samples (str, int): Number of files that will returned in total (summing training and validation)

    Returns:
    """
    random_state = 42
    data = pd.read_csv(csv_file)
    # split the data into positive and negative examples. Since we have less positive examples, we can reduce the amount
    # of negative ones
    df_neg = data[data['target'] == 0]
    df_pos =data[data['target'] == 1]

    if num_samples == 'all':
        num_samples = len(df_neg)
    else:
        if num_samples > len(df_neg):
            num_samples = len(df_neg)
        else:
            pass
    df_neg = df_neg.sample(num_samples, random_state=random_state)

    #concatenate data
    out = pd.concat([df_neg, df_pos]).reset_index()
    out['images'] = out['image_id'].apply(lambda name: os.path.join(image_path, name + '.jpg'))
    return out[['images', 'target']]

def create_datasets(dataframe, test_size: float = 0.2, train_transform = None, test_transform = None, random_state = 42):
    x_train, x_test, y_train, y_test = train_test_split(dataframe['images'], dataframe['target'],
                                                      test_size=test_size, random_state=random_state)

    train_dataset = data_loader.ISICDataset(data=x_train.values, label=y_train.values, transform=train_transform)
    test_dataset = data_loader.ISICDataset(data=x_test.values, label=y_test.values, transform=test_transform)

    return train_dataset, test_dataset

class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha: float=0.25, gamma: float =2.0, reduce="none", n_classes=1):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Implementation based on https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        Args:
            alpha (float): weighting factor between (0, 1) to balance positive vs negative examples.
            -1 will not weight
            gamma (float): Exponent of the modulating factor (1 - pt) to balance easy vs hard examples
            reduce (str): 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.n_classes = n_classes

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        targets = F.one_hot(targets, self.n_classes).float()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduce == "mean":
            loss = loss.mean()
        elif self.reduce == "sum":
            loss = loss.sum()

        return loss

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = create_dataframe(r'C:\Users\15B38LA\Downloads\ISIC_2020\512x512-dataset-melanoma\512x512-dataset-melanoma',
                     r'C:\Users\15B38LA\Downloads\ISIC_2020\marking.csv',
                     6000)
    train_ds, test_ds = create_datasets(test)

    for item in range(len(train_ds)):
        c, d = train_ds[item]
        plt.imshow(c)
