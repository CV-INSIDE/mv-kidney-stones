import os
import shutil
import skimage.io as io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

"""
This class handles all the methods related to image manipulation.
"""
class ImageManager(object):
  def __init__(self, source_path, target_path):
    self.target_path = target_path
    self.source_path = source_path


  """
  Method to split the dataset into train and test sets.
  """
  def prepare_data(self, test_percentage=0.2, merge_classes=None):
    if merge_classes != None:
      self._prepare_all_merge(test_percentage=test_percentage, merge_classes=merge_classes)
    else:
      self._prepare_all_no_merge(test_percentage=test_percentage)

  """
  Creates the "train" and "test" folders and split the files into these folders.
  """
  def _copy_files(self, store_path, images_per_cat_arr, test_percentage):
      for cat in images_per_cat_arr:
        x = images_per_cat_arr[cat]
        train_x, val_x, train_y, val_y = train_test_split(x,
                                      x, test_size = test_percentage, shuffle = True)
        print("Distribution for %s => train: %s, test: %s" % (cat, len(train_x), len(val_x)))

        target_dir = store_path + "/train/" + cat
        for filename in train_x:
          shutil.copy2(filename, target_dir)
        target_dir = store_path + "/test/" + cat
        for filename in val_x:
          shutil.copy2(filename, target_dir)

  """
  Reads all the images (surface and cross section) and split them
  into "train" and "test" folders according to the percentage given as input.
  """
  def _prepare_all_no_merge(self, test_percentage):
    store_path = self.target_path
    if os.path.exists(store_path):
      shutil.rmtree(store_path)

    os.mkdir(store_path)
    os.mkdir(store_path + "/train")
    os.mkdir(store_path + "/test")
    # read every image and store them in an array.
    images_per_cat = { }
    for subdir, dirs, files in os.walk(self.source_path):
      category = subdir.lower().replace(self.source_path, "")
      for file in files:
        img_path = subdir + "/" + file
        #img = io.imread(img_path)
        if category not in images_per_cat:
          images_per_cat[category] = []
          os.mkdir(store_path + "/train/" + category)
          os.mkdir(store_path + "/test/" + category)

        images_per_cat[category].append(img_path)

    self._copy_files(store_path, images_per_cat, test_percentage)

  """
  Reads all the images (surface and cross section) and reorganize them
  into a single category (WEDDELLITE, WHEWELLITE or ACIDE) and split them
  into "train" and "test" folders according to the percentage given as input.
  """
  def _prepare_all_merge(self, test_percentage, merge_classes):
    store_path = self.target_path
    if os.path.exists(store_path):
      shutil.rmtree(store_path)

    os.mkdir(store_path)
    os.mkdir(store_path + "/train/")
    os.mkdir(store_path + "/test/")
    # read every image and store them in an array.
    images_per_cat = { }
    for subdir, dirs, files in os.walk(self.source_path):
      category = 'unknown'
      if len(files) > 0:
        for cat_key, folders in merge_classes.items():
          for folder_name in folders:
            if folder_name in subdir.lower():
              category = cat_key

      for file in files:
        img_path = subdir + "/" + file
        #img = io.imread(img_path)
        if category not in images_per_cat:
          images_per_cat[category] = []
          os.mkdir(store_path + "/train/" + category)
          os.mkdir(store_path + "/test/" + category)

        images_per_cat[category].append(img_path)

    self._copy_files(store_path, images_per_cat, test_percentage)
