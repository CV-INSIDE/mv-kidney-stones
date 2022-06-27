"""
Helper function created for multi-view model. Observes the IDs from surface and section images and keep only those that
are shared.
"""
import os
import numpy as np


# section_folder = r'C:\Users\15B38LA\Downloads\section_original'
# surface_folder = r'C:\Users\15B38LA\Downloads\surface_original'

section_folder = r'C:\Users\15B38LA\Downloads\surface\surface\test'
surface_folder = r'C:\Users\15B38LA\Downloads\surface\surface\train'


def get_content_of_folder_original(wd: str) -> dict:
    """
    Get the contents of folder wd and returns a dictionary with all the elements contained in each sub folder
    :param wd: path to the folder of interest
    :return: a dictionary containing all the elements found in wd
    """
    out = {}
    for root, dirs, files in os.walk(wd):
        for name in files:
            # append image path to the folder
            image_id = get_image_id(name)
            if image_id not in out[os.path.basename(root)].keys():
                out[os.path.basename(root)][get_image_id(name)] = []
                out[os.path.basename(root)][get_image_id(name)].append(os.path.join(root, name))
            else:
                out[os.path.basename(root)][get_image_id(name)].append(os.path.join(root, name))
        for name in dirs:
            # create an entry in the dictionary
            out[name] = {}

    return out

def get_content_of_folder(wd: str):
    """
    Get the contents of folder wd and returns a dictionary with all the elements contained in each sub folder
    :param wd: path to the folder of interest
    :return: a dictionary containing all the elements found in wd
    """
    out = []
    labels = []
    for root, dirs, files in os.walk(wd):
        for name in files:
            # append image path to the folder
            if not name.endswith('.zip'):
                out.append(os.path.join(root,name))
                labels.append(os.path.basename(root))
            else:
                pass
    return out, labels


def get_image_id(img_path: str):
    """
    This function only works when image starts with the ID of the trial
    The format of the images must be: numeric id - section/surface + additional information
    :return:
    """
    img_path = img_path.split('-')[0]
    img_path = img_path.strip()
    return img_path


def compare_folders(data1: dict, data2: dict):
    # First, assure that the keys from both inputs are the same, otherwise it would not be necessary to do this
    if not (data1.keys() == data2.keys()):
        raise ValueError('Mismatching input values')

    out = {}
    total = 0
    for key in data1.keys():
        # get subkeys. Use the bigger dataset
        experiments_data = data1[key].keys() if len(data1[key].keys()) > len(data2[key].keys()) else data2[key].keys()
        secondary_data = data1[key].keys() if len(data1[key].keys()) < len(data2[key].keys()) else data2[key].keys()
        out[key] = {}
        for experiment in experiments_data:
            if experiment in secondary_data:
                out[key][experiment] = min(len(data1[key][experiment]), len(data2[key][experiment]))
                total += out[key][experiment]
    return out


def compare_folders_test(data1: dict, data2: dict):
    # First, assure that the keys from both inputs are the same, otherwise it would not be necessary to do this
    if not (data1.keys() == data2.keys()):
        raise ValueError('Mismatching input values')

    out = {}
    total = 0
    for key in data1.keys():
        # get subkeys. Use the bigger dataset
        experiments_data = data1[key].keys() if len(data1[key].keys()) > len(data2[key].keys()) else data2[key].keys()
        secondary_data = data1[key].keys() if len(data1[key].keys()) < len(data2[key].keys()) else data2[key].keys()
        out[key] = {}
        for experiment in experiments_data:
            try:
                for aaa in data1[key][experiment]:
                    if aaa in data2[key][experiment]:
                        print("Si")
            except KeyError:
                pass
    return out

def balance(data1: dict, data2: dict, reference: dict):
    """
    Match the number of images present in each dataset based on the minimum number of images that was available
    """
    balanced_data1 = []
    balanced_data2 = []
    for key in reference.keys():
        for label, qty in reference[key].items():
            balanced_data1 += data1[key][label][0:qty]
            balanced_data2 += data2[key][label][0:qty]

    return balanced_data1, balanced_data2


def shuffle_data(data1: list):
    """
    Returns a list of random indices that will be used to create the train and validation dataset
    :param data1:
    :return:
    """
    dataset_size = len(data1)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    return indices


def split_data(data1, data2, index, percentage=0.2):
    """
    Create two folders, one for data1 and one for data2. Inside the folders, data will be split into two datasets.
    One for training and one for test. Additionally, a folder containing their class type is also created.
    :param data1:
    :param data2:
    :param index:
    :return:
    """
    if percentage < 0.0 or percentage > 1.0:
        raise ValueError('Invalid percentage Value')

    if len(data1) != len(data2):
        raise ValueError('Invalid data size!')

    split_index = int(np.floor(percentage * len(data1)))
    train, test = index[split_index:], index[:split_index]

    hola = 1


if __name__ == '__main__':
    section = get_content_of_folder(section_folder)
    surface = get_content_of_folder(surface_folder)
    for img in section:
        if img in surface:
            print("si")

    for img in surface:
        if img in section:
            print("si")
    compared_folders = compare_folders_test(section, surface)
    balanced1, balanced2 = balance(surface, section, compared_folders)
    split_data(balanced1, balanced2, shuffle_data(balanced1))
    hola = 1


