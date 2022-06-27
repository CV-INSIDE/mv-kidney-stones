import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, utils
import seaborn as sns
import skimage.io as io

from sklearn.metrics import roc_curve,auc
from scipy import interp
from itertools import cycle

#import sklearn.datasets
#import umap
#from sklearn import preprocessing
#from bokeh.plotting import show, save, output_notebook, output_file
#import umap.plot

import scprep
import umap
import sklearn.manifold

"""
This class handles all the methods related to plots.
"""
class PlotHelper:
  def __init__(self):
    pass

    """
    Displays an image. It receives as input the image path.
    """
    def show_image_from_path(self, imgPath):
      im = io.imread(imgPath)
      plt.imshow(im)

    """
    Reads the first X images from the given path and displays them in an row.
    """
    def plot_random_images(self, img_path, num_images=10):
      images_arr = []
      images_names = []
      for subdir, dirs, files in os.walk(img_path):
          max_files = files[:num_images]
          for file in max_files:
            img_path = subdir + "/" + file
            img = io.imread(img_path)
            images_arr.append(img)
            images_names.append(file)

      fig, axes = plt.subplots(1, num_images, figsize=(35, 35))
      axes = axes.flatten()
      i = 0
      for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.set_title(images_names[i])
        i = i + 1
      plt.tight_layout()
      plt.show()

  """
  Returns the image as a tensor.
  """
  def image_as_tensor(self, img_path):
    transformations = [ transforms.ToTensor() ]
    def image_loader(loader, image_name):
        image = Image.open(image_name)
        image = loader(image).float()
        image = torch.tensor(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image

    data_transforms = transforms.Compose(transformations)

    im_loader = image_loader(data_transforms, img_path)
    im =  next(iter(im_loader))
    return im     
      

  """
  Applies the passed array of transformations to the given image.
  This method is to test transformations on a single image.
  """
  def test_transformations_on_image(self, img_path, transformations):
    def image_loader(loader, image_name):
        image = Image.open(image_name)
        image = loader(image).float()
        image = torch.tensor(image, requires_grad=True)
        image = image.unsqueeze(0)
        return image

    data_transforms = transforms.Compose(transformations)

    im_loader = image_loader(data_transforms, img_path)
    im =  next(iter(im_loader))
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(io.imread(img_path))
    axarr[1].imshow(transforms.ToPILImage()(im), interpolation="bicubic")

  """
  Plots the classes distribution for the given dataset.
  """
  def plot_dataset_class_distribution(self, dataset_obj, idx2class, plot_title="Entire dataset", **kwargs):
    plt.figure(figsize=(15, 8))
    count_dict = { k: 0 for k,v in dataset_obj.class_to_idx.items() }
    for _, label_id in dataset_obj:
      label = idx2class[label_id]
      count_dict[label] += 1

    sns.barplot(data = pd.DataFrame.from_dict([count_dict]).melt(),
    x="variable", y="value", hue="variable", **kwargs).set_title(plot_title)

  """
  Plots the classes distribution for the given data loader.
  """
  def plot_loader_class_distribution(self, dataloader_obj, dataset_obj, plot_title="Entire dataset", **kwargs):
    plt.figure(figsize=(15, 8))
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    if dataloader_obj.batch_size == 1:
        for _,label_id in dataloader_obj:
            y_idx = label_id.item()
            y_lbl = idx2class[y_idx]
            count_dict[str(y_lbl)] += 1
    else:
        for _,label_id in dataloader_obj:
            for idx in label_id:
                y_idx = idx.item()
                y_lbl = idx2class[y_idx]
                count_dict[str(y_lbl)] += 1

    sns.barplot(data = pd.DataFrame.from_dict([count_dict]).melt(),
    x="variable", y="value", hue="variable", **kwargs).set_title(plot_title)


  # shows some images from the loader.
  def print_loader(self, loader):
    single_batch = next(iter(loader))
    single_batch_grid = utils.make_grid(single_batch[0], nrow=4)
    plt.figure(figsize = (10,10))
    plt.imshow(single_batch_grid.permute(1, 2, 0))

  """
  Displays the accuracy and loss graphs.
  """
  def show_training_results(self, accuracy_history, loss_history, logger=None):
    train_val_acc_df = pd.DataFrame.from_dict(accuracy_history).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
    train_val_loss_df = pd.DataFrame.from_dict(loss_history).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

    # Plot line charts
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
    sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
    sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')

    if logger:
      img = fig.savefig("dummy.png")
      my_img = self.image_as_tensor(img_path="dummy.png")
      logger.experiment.add_image('Accuracy and Loss', my_img)
    
  """
  Displays the ROC plot for the given data and classes.
  """
  def plot_roc(self, y_data, y_pred_data, n_classes, title='ROC plot', labels=None):
    y_test = y_data
    pred1 = y_pred_data
    
    if not labels:
      labels = {}
      for i in range(n_classes):
        labels[i] = str(i)

    t1=sum(x==0 for x in pred1-y_test)/len(pred1)

    ### MACRO
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(pred1))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'magenta', 'navy', 'forestgreen', 'gray', 'sienna', 'gold'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.48),color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    
  """
  Plots umap
  """

  def umap_plot(self, features_df, target_col, sample_size=None, labels=None, knn_neighbors=15, min_dist=0.1, dim='2d', random_state=None):
    x = features_df.drop(target_col, axis=1)

    if labels:
      y = features_df[target_col].map(labels)
    else:
      y = features_df[target_col]

    if not sample_size:
      sample_size = len(x)

    # apply pca to select the top 100 features and plot the graph faster.
    max_features = min(len(x.columns), 100)
    x_pca = scprep.reduce.pca(x, n_components=max_features, method='dense')
    x, y = scprep.select.subsample(x_pca, y, n=sample_size, seed=random_state)

    if dim == '3d':
      umap_op = umap.UMAP(n_components=3, n_neighbors=knn_neighbors,
                          min_dist=min_dist, random_state=random_state)
      data_umap = umap_op.fit_transform(x)
      scprep.plot.scatter3d(data_umap, c=y,
                          figsize=(8,4), legend_anchor=(1,1), ticks=False, label_prefix='umap_')
    else:
      umap_op = umap.UMAP(n_components=2, n_neighbors=knn_neighbors,
                          min_dist=min_dist, random_state=random_state)
      data_umap = umap_op.fit_transform(x)
      scprep.plot.scatter2d(data_umap, c=y,
                          figsize=(8,4), legend_anchor=(1,1), ticks=False, label_prefix='umap_')



  """
  Plots tsne
  """

  def tsne_plot(self, features_df, target_col, sample_size=None, labels=None, dim='2d', random_state=None,
               perplexity=30,
              learning_rate=200,
              n_iter=1000,
              n_iter_without_progress=300,
              min_grad_norm= 1e-7,
              init="random",
              method="barnes_hut",
              angle=0.5,
              n_jobs=None,
              early_exaggeration=12.0):
    x = features_df.drop(target_col, axis=1)

    if labels:
      y = features_df[target_col].map(labels)
    else:
      y = features_df[target_col]

    if not sample_size:
      sample_size = len(x)

    # apply pca to select the top 100 features and plot the graph faster.
    max_features = min(len(x.columns), 100)
    x_pca = scprep.reduce.pca(x, n_components=max_features, method='dense')
    x, y = scprep.select.subsample(x_pca, y, n=sample_size, seed=random_state)

    if dim == '3d':
      tsne_op = sklearn.manifold.TSNE(n_components=3, random_state=random_state,
              perplexity= perplexity,
              learning_rate= learning_rate,
              n_iter= n_iter,
              n_iter_without_progress= n_iter_without_progress,
              min_grad_norm= min_grad_norm,
              init=init,
              method=method,
              angle=angle,
              n_jobs=n_jobs)
      data_tsne = tsne_op.fit_transform(x)
      scprep.plot.scatter3d(data_tsne, c=y,
                          figsize=(8,4), legend_anchor=(1,1), ticks=False, label_prefix='tsne_')
    else:
      tsne_op = sklearn.manifold.TSNE(n_components=2, random_state=random_state,
              perplexity= perplexity,
              learning_rate= learning_rate,
              n_iter= n_iter,
              n_iter_without_progress= n_iter_without_progress,
              min_grad_norm= min_grad_norm,
              init=init,
              method=method,
              angle=angle,
              n_jobs=n_jobs)
      data_tsne = tsne_op.fit_transform(x)
      scprep.plot.scatter2d(data_tsne, c=y,
                          figsize=(8,4), legend_anchor=(1,1), ticks=False, label_prefix='tsne_')
