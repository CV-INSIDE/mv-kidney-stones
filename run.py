import os
import argparse
import yaml

import numpy as np
import torch
from torchvision import transforms
import random
import pytorch_lightning as pl
from collections import OrderedDict

from pytorch_lightning.callbacks import EarlyStopping

import helpers.ks_imageloader_mv as kl


def main(config):
    TOTAL_GPUS = 0
    if torch.cuda.is_available():
        TOTAL_GPUS = 1

    mean = []
    std = []

    for val in config['image_mean'].split(","):
        mean.append(float(val))

    for val in config['image_std'].split(","):
        std.append(float(val))

    mean = np.array(mean)
    std = np.array(std)

    if config['model_to_use'] == "alexnet":
        from models.alexnet import AlexnetModel
        model = AlexnetModel(hparams={"lr": 0.0002}, num_classes=config['num_classes'], pretrained=True, seed=config['manualSeed'])
        IMG_SIZE = 227
    elif config['model_to_use'] == "multiview":
        from models.multiview import MultiViewMaxPool
        model = MultiViewMaxPool(hparams={"lr": 0.0002}, num_classes=config['num_classes'], pretrained=True, seed=config['manualSeed'])
        # model = MultiView.load_from_checkpoint(checkpoint_path=
        # r'C:\Users\15B38LA\Downloads\mixed_kidney_yelbeze.ckpt', strict=False)
        checkpoint = torch.load(r'C:\Users\15B38LA\Downloads\mixed_kidney_yelbeze.ckpt', map_location=lambda storage, loc: storage)
        test = OrderedDict({k: v for k, v in checkpoint['state_dict'].items() if 'classifier' not in k})
        model.load_state_dict(test, strict=False)
    elif config['model_to_use'] == "mv_vgg16_max":
        from models.multiview import MultiViewPoolVGG16
        model = MultiViewPoolVGG16(hparams={"lr": 0.00005}, num_classes=config['num_classes'], pretrained=True,
                                 seed=config['manualSeed'])
        # model = MultiView.load_from_checkpoint(checkpoint_path=
        # r'C:\Users\15B38LA\Downloads\mixed_kidney_yelbeze.ckpt', strict=False)
        checkpoint = torch.load(r'C:\Users\15B38LA\Documents\vgg16-mixed.ckpt',
                                map_location=lambda storage, loc: storage)
        test = OrderedDict({k: v for k, v in checkpoint['state_dict'].items() if 'classifier' not in k})
        model.load_state_dict(test, strict=False)
        IMG_SIZE = 227
    elif config['model_to_use'] == "inception":
        from models.inception import InceptionModel
        model = InceptionModel(hparams={"lr": 0.0002}, num_classes=config['num_classes'], pretrained=True, seed=config['manualSeed'])
        IMG_SIZE = 224
    elif config['model_to_use'] == "inception_mv":
        from models.inception import InceptionModeMulti
        model = InceptionModeMulti(hparams={"lr": 0.0002}, num_classes=config['num_classes'], pretrained=True, seed=config['manualSeed'])
        checkpoint = torch.load(r'C:\Users\15B38LA\Documents\inception_mixed.ckpt',
                                map_location=lambda storage, loc: storage)
        test = OrderedDict({k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k})
        model.load_state_dict(test, strict=False)
        IMG_SIZE = 224
    elif config['model_to_use'] == "vgg19":
        from models.vgg19 import Vgg19Model
        model = Vgg19Model(hparams={"lr": 0.00005}, num_classes=config['num_classes'], pretrained=True, seed=config['manualSeed'])
        IMG_SIZE = 224
    elif config['model_to_use'] == "vgg16":
        from models.vgg16 import Vgg16Model
        model = Vgg16Model(hparams={"lr": 0.00005}, num_classes=config['num_classes'], pretrained=True, seed=config['manualSeed'])
        IMG_SIZE = 224
    else:
        raise ValueError('Model is not implemented')

    # Transformations
    random_transforms = 1
    train_transformations = []
    train_transformations += [transforms.RandomChoice([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Pad(50, fill=0, padding_mode="symmetric"),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.5),
        transforms.RandomAffine(degrees=(-90, 90), translate=(0, 0.2), scale=[0.5, 1]),
        # transforms.ColorJitter(brightness=0.35, contrast=0.4, saturation=0.5, hue=0),
        transforms.RandomRotation(degrees=(-180, 180)),
    ]) for _ in range(random_transforms)]
    train_transformations += [transforms.Resize((IMG_SIZE, IMG_SIZE)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean, std)]
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
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)   
    ]

    if config['are_images_gray'] == "yes":
        train_transformations.insert(0, transforms.Grayscale(num_output_channels=3))
        test_transformations.insert(0, transforms.Grayscale(num_output_channels=3))

    # WITH augmentation
    image_transforms = {
        "train": transforms.Compose(train_transformations),
        "test": transforms.Compose(test_transformations)
    }

    # WITHOUT augmentation
    image_transforms_no_aug = {
        "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)                           
        ]),
        "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

    # if config['image_path']:
    #    base_test_path = config['image_path'] + "/test/"

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


    # Variable that holds the route to the image zip files. Change this value if
    # you wish to run the test on a different set of images. IMPORTANT: The zip file
    # must contain images ordered in the same structure ("train" and "test" folders).

    loader = kl.KidneyImagesLoader(# images_path=config['image_path'],
                            images_path=[config['image_path_view1'], config['image_path_view2']],
                            val_percentage=0.2,
                            train_batch_size = 1,
                            train_transformations=image_transforms["train"],
                            test_transformations=image_transforms["test"],
                            seed=config['manualSeed'])
    pl.seed_everything(config['manualSeed'])


    trainer = pl.Trainer(gpus=None,
                        max_epochs=config['max_exec_epochs'],
                        min_epochs=config['min_exec_epochs'],
                        #logger=logger,
                        #callbacks=[stopping],]
                        progress_bar_refresh_rate=50,
                        checkpoint_callback=False, # disable checkpoint logs
                        #auto_lr_find=True,
                        deterministic= True
                        )

    print('### Model: ###')
    print(model)

    #trainer.fit(model, loader)
    # trainer.save_checkpoint(f"saves/{config['model_name']}.ckpt")
    model2 = MultiViewPoolVGG16.load_from_checkpoint(r'C:\Users\15B38LA\Documents\mv_vgg16_max.ckpt')
    trainer.test(model2, datamodule=loader)
    hola = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--yaml', default='config.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.yaml, 'r'), Loader=yaml.FullLoader)
    print('Setting gpu: ', args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(config)