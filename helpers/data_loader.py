"""
Custom DataLoader used to create images in HSV and LBP
"""
import os

import numpy as np
import torch
import pandas as pd

from torch.utils.data import Dataset
from skimage import io
from PIL import Image

import helpers.transformations as transformations
# from helpers.data_separator import get_content_of_folder

# import matplotlib.pyplot as plt


class ColorDataSet(Dataset):
    """
    Kidney Color Transformation dataset
    """
    def __init__(self, data, transform=None, hsv: bool = False, lbp: bool = False,
                 select_class=None, select_color=None, train=False):
        """
        Args:
            data: dictionary/dataframe containing the images to parse
            hsv: if True, hsv transformation will be applied to the images
            lbp: if True, lbp transformation will be applied to the images
        """
        if select_class is None:
            select_class = []

        if select_color is None:
            select_color = []

        self.transform = transform
        self.hsv = hsv
        self.lbp = lbp
        self.data = pd.DataFrame()
        if train:
            self._apply_transformations(pd.DataFrame(data={'image': data[0],
                                                           'labels': data[1],
                                                           'color': ['rgb'] * len(data[1])}))
        else:
            self.data = pd.DataFrame(data={'image': data[0], 'labels': data[1], 'color': ['rgb'] * len(data[1])})
        self.class_to_idx = {klass: i for i, klass in enumerate(self.data['labels'].unique())}

        # Filter by class and by color config
        if select_class and len(select_class) > 0:
            self.data = pd.concat([self.data[self.data['labels'] == lab] for lab in select_class])

        if select_color and len(select_color) > 0:
            self.data = pd.concat([self.data[self.data['color'] == lab] for lab in select_color])

    def _apply_transformations(self, df):
        """
        Create an entry in a dataframe depending if hsv or lbp is required. Otherwise a single rgb image will be passed
        """
        tmp_img, tmp_lab, tmp_color = [], [], []

        for idx in range(len(df)):
            # add an entry for the rgb original image
            image_name = os.path.abspath(df.iloc[idx, 0])
            label = df.iloc[idx, 1]

            tmp_img.append(image_name)
            tmp_lab.append(label)
            tmp_color.append(df.iloc[idx, 2])

            if self.hsv:
                tmp_img.append(image_name)
                tmp_lab.append(label)
                tmp_color.append('hsv')

            if self.lbp:
                tmp_img.append(image_name)
                tmp_lab.append(label)
                tmp_color.append('lbp')

        self.data = pd.DataFrame(data={'image': tmp_img, 'labels': tmp_lab, 'color': tmp_color})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.abspath(self.data.iloc[idx, 0])
        image = io.imread(image_name)
        mode = 'RGB'

        if self.data.iloc[idx, 2] == 'hsv':
            image = transformations.transform_hsv(image)
            mode = 'HSV'

        elif self.data.iloc[idx, 2] == 'lbp':
            image, _ = transformations.transform_lbp(image)
            image = np.stack([image, image, image])
            image = np.moveaxis(image, 0, -1)

        image = Image.fromarray(image, mode=mode)
        if self.transform:
            image = self.transform(image)

        # sample = {'image': image,  'label': self.data.iloc[idx, 1]}
        return image, self.class_to_idx[self.data.iloc[idx, 1]]


class ColorDataSetMV(ColorDataSet):
    """
    Dataset used for multiview validation. __getitem__  method will return a view for RGB, a view for HSV, and a view
    for LBP por each image.
    """
    def __init__(self, data, transform=None, select_class=None, select_color=None):
        super().__init__(data, transform, hsv=False, lbp=False, select_class=select_class,
                         select_color=select_color, train=False)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = os.path.abspath(self.data.iloc[idx, 0])
        # image in RGB
        image_1 = io.imread(image_name)

        # get image in hsv
        image_2 = transformations.transform_hsv(image_1)

        # get image in lbp
        image_3, _ = transformations.transform_lbp(image_1)
        image_3 = np.stack([image_3, image_3, image_3])
        image_3 = np.moveaxis(image_3, 0, -1)

        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(image_1)
        # plt.subplot(1,3,2)
        # plt.imshow(image_2)
        # plt.subplot(1,3,3)
        # plt.imshow(image_3)

        # transform images into PIL format
        image_1 = Image.fromarray(image_1, mode='RGB')
        image_2 = Image.fromarray(image_2, mode='HSV')
        image_3 = Image.fromarray(image_3, mode='RGB')

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image_3 = self.transform(image_3)

        return (image_1, image_2, image_3), self.class_to_idx[self.data.iloc[idx, 1]]


# test = r'C:\Users\15B38LA\Downloads\mixed\test'
# data = get_content_of_folder(test)
# ewe = ColorDataSet(data, hsv=True, lbp=True, train=True, select_color=['hsv'])
# ewe = ColorDataSetMV(data)
# for item in range(len(ewe)):
#     a, b = ewe[item]
