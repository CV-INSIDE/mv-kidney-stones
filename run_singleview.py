import os
import yaml
import torch
import random
import logging
import argparse

import numpy as np
from datetime import datetime
import pytorch_lightning as pl
import helpers.ks_imageloader_single as kl

from torchvision import transforms
from collections import OrderedDict
from pytorch_lightning.callbacks import EarlyStopping

# logging details
now =datetime.now()
current_time =now.strftime("%H%M%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler(f"single_view_{current_time}.log")
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# logging.basicConfig(filename=f"single_view_{current_time}.log", format='%(asctime)s - %(message)s', level=logging.INFO)

def main(config):
    """
    Main process
    """

    # Extract information from configuration file.
    # The first extracted information is the dataset mean and standard deviation that will be used for the transformation
    mean, std = [], []
    img_size = 227

    for val in config['image_mean'].split(","):
        mean.append(float(val))

    for val in config['image_std'].split(","):
        std.append(float(val))

    mean = np.array(mean)
    std = np.array(std)

    # Accepted models are:
    # Single view models:
    #   * alexnet, inception, vgg16, vgg19
    # Multi view models (require a pretrained model):
    #   * alexnet_mv_max, vgg16_mx_max, inception_mv_max

    # Single View
    if config['model_to_use'] == "alexnet":
        logging.info("Using alexnet model")
        from models.alexnet import AlexnetModel
        model = AlexnetModel(hparams={"lr": 0.0002}, num_classes=config['num_classes'],
                             pretrained=True, seed=config['manualSeed'])
        img_size = 227
    elif config['model_to_use'] == "resnet50":
        logging.info("Using resnet50 model")
        from models.attention.models.resnet import ResNet50Cbam
        model = ResNet50Cbam(hparams={"lr": 0.0002}, num_classes=config['num_classes'],
                             seed=config['manualSeed'])
        img_size = 224
    elif config['model_to_use'] == "inception":
        logging.info("using inception model")
        from models.inception import InceptionModel
        model = InceptionModel(hparams={"lr": 0.0002}, num_classes=config['num_classes'],
                               pretrained=True, seed=config['manualSeed'])
        img_size = 224
    elif config['model_to_use'] == "vgg16":
        logging.info("using vgg16 model")
        from models.vgg16 import Vgg16Model
        model = Vgg16Model(hparams={"lr": 0.00005}, num_classes=config['num_classes'],
                           pretrained=True, seed=config['manualSeed'])
        img_size = 224
    elif config['model_to_use'] == "vgg19":
        logging.info("using vgg19 model")
        from models.vgg19 import Vgg19Model
        model = Vgg19Model(hparams={"lr": 0.00005}, num_classes=config['num_classes'],
                           pretrained=True, seed=config['manualSeed'])
        img_size = 224

    # multi view
    elif config['model_to_use'] == "alexnet_mv_max":
        logging.info("using multiview alexnet max model")
        from models.multiview import MultiViewMaxPool
        model = MultiViewMaxPool(hparams={"lr": 0.0002}, num_classes=config['num_classes'],
                                 pretrained=True, seed=config['manualSeed'])

        checkpoint = torch.load(r'C:\Users\15B38LA\Downloads\mixed_kidney_yelbeze.ckpt',
                                map_location=lambda storage, loc: storage)
        test = OrderedDict({k: v for k, v in checkpoint['state_dict'].items() if 'classifier' not in k})
        model.load_state_dict(test, strict=False)
        img_size = 227
    elif config['model_to_use'] == "vgg16_mv_max":
        logging.info("using multiview vgg16 max model")
        from models.multiview import MultiViewPoolVGG16
        model = MultiViewPoolVGG16(hparams={"lr": 0.00005}, num_classes=config['num_classes'], pretrained=True,
                                   seed=config['manualSeed'])
        checkpoint = torch.load(r'C:\Users\15B38LA\Documents\vgg16-mixed.ckpt',
                                map_location=lambda storage, loc: storage)
        test = OrderedDict({k: v for k, v in checkpoint['state_dict'].items() if 'classifier' not in k})
        model.load_state_dict(test, strict=False)
        img_size = 227
    elif config['model_to_use'] == "inception_mv_max":
        logging.info("using multi view inception max model")
        from models.inception import InceptionModeMulti
        model = InceptionModeMulti(hparams={"lr": 0.0002}, num_classes=config['num_classes'], pretrained=True,
                                   seed=config['manualSeed'])

        checkpoint = torch.load(r'C:\Users\15B38LA\Documents\inception_mixed.ckpt',
                                map_location=lambda storage, loc: storage)
        test = OrderedDict({k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k})
        model.load_state_dict(test, strict=False)
        img_size = 224

    # default case
    else:
        raise ValueError('Model is not implemented')

    # Transformations
    random_transforms = 1
    train_transformations = [transforms.ToTensor()]
    train_transformations += [transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Pad(50, fill=0, padding_mode="symmetric"),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2), scale=[0.5, 1]),
        # transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.5, hue=0),
        transforms.RandomRotation(degrees=(-180, 180)),
    ]) for _ in range(random_transforms)]
    train_transformations += [transforms.Resize((img_size, img_size)),
                              ]
                              #transforms.Normalize(mean, std)]
    """
    transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Pad(50, fill=0, padding_mode="symmetric"),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2), scale=[0.5, 1]),
        # transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.5, hue=0),
        transforms.RandomRotation(degrees=(-180, 180)),
    ]),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)                           
    ]
    """
    test_transformations = [
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        #transforms.Normalize(mean, std)
    ]

    if config['are_images_gray'] == "yes":
        train_transformations.insert(0, transforms.Grayscale(num_output_channels=3))
        test_transformations.insert(0, transforms.Grayscale(num_output_channels=3))


    if config['use_augmentation']:
        # with augmentation
        image_transforms = {
            "train": transforms.Compose(train_transformations),
            "test": transforms.Compose(test_transformations)
        }

    else:
        # without augmentation
        image_transforms = {
            "train": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            "test": transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    # Early stopping
    stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=30,
        verbose=False,
        mode='min'
    )

    if config['manualSeed'] != None:
        manualSeed = config['manualSeed']
        np.random.seed(manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        pl.seed_everything(manualSeed)
        os.environ['PYTHONHASHSEED'] = str(manualSeed)

    print("STD => ", std)
    print("MEAN => ", mean)
    print("IMAGE TRANSFORMATIONS => ", image_transforms)

    logging.info(f"transformations: {image_transforms}")

    # Variable that holds the route to the image zip files. Change this value if
    # you wish to run the test on a different set of images. IMPORTANT: The zip file
    # must contain images ordered in the same structure ("train" and "test" folders).

    loader = kl.KidneyImagesLoader(images_path=config['image_path'],
                                   val_percentage=0.2,
                                   train_batch_size=16,
                                   train_transformations=image_transforms["train"],
                                   test_transformations=image_transforms["test"],
                                   seed=config['manualSeed'],
                                   color_transform=config['color_transform'][0],
                                   hsv=config['color_transform'][1],
                                   lbp=config['color_transform'][2])
    pl.seed_everything(config['manualSeed'])

    # Class
    logging.info(f"min epochs: {config['min_exec_epochs']}")
    logging.info(f"max epochs: {config['max_exec_epochs']}")
    trainer = pl.Trainer(gpus=None,
                         max_epochs=config['max_exec_epochs'],
                         min_epochs=config['min_exec_epochs'],
                         # logger=logger,
                         # callbacks=[stopping],]
                         progress_bar_refresh_rate=50,
                         checkpoint_callback=False,  # disable checkpoint logs
                         # auto_lr_find=True,
                         deterministic=True
                         )

    print('### Model: ###')
    print(model)
    logging.info(f"model: {model}")

    trainer.fit(model, loader)
    trainer.test(model, datamodule=loader)
    if config['save_model']:
        trainer.save_checkpoint(f"saves/{config['model_name']}.ckpt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='cuda')
    parser.add_argument('--yaml', default='config.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.yaml, 'r'), Loader=yaml.FullLoader)
    # Setting device if cuda is available
    cuda = torch.cuda.is_available()
    device = torch.device(args.gpu if cuda else 'cpu')
    # start the main process
    main(config)
