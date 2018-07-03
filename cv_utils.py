import numpy as np
import traceback
from torchvision import transforms, datasets, utils, models
from torch.utils.data import DataLoader
from torch import device, cuda
import os
import matplotlib.pyplot as plt
import time
from torch import nn

def get_data_transforms(ccmean, ccstd, crop_size, 
                            data_types=['train', 'test', 'val']):
    try:
        if len(ccmean) != len(ccstd) or len(ccmean) != len(data_types):
            return None
        data_transforms = {k:transforms.Compose([
                                transforms.RandomResizedCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(ccmean, ccstd)
                            ]) for k in data_types}
        return data_transforms
    except Exception as e:
        print(traceback.format_exc())
        raise e


def get_dataloader_obj(data_dir, data_transforms, 
                        data_types=['train', 'test', 'val'], bs=4):
    try:
        image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                            for x in data_types}
        dataloaders = {x:DataLoader(image_datasets[x], batch_size=bs,
                            shuffle=True, num_workers=bs) for x in data_types}
        dsizes = {x:len(image_datasets[x]) for x in data_types}
        class_names = image_datasets['train'].classes
        dev = device("cuda:0" if cuda.is_available() else "cpu")
        return dataloaders, dsizes, class_names, dev
    except Exception as e:
        print(traceback.format_exc())
        raise e

def imshow(inp, ccmean, ccstd, title=None):
    inp = utils.make_grid(inp)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(ccmean)
    std = np.array(ccstd)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.imshow(inp)
    plt.pause(5)
    plt.close()


def get_model(mname, num_class, dev, pretrained=True):
    try:
        model_ft = None
        if mname == 'resnet18' and pretrained:
            model_ft = models.resnet18(pretrained=True)
        elif mnane == 'resnet18' and not pretrained:
            model_ft = models.resnet18()
        
        if model_ft:
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_class)
            model_ft = model_ft.to(dev)
            return model_ft
        return model_ft
    except Exception as e:
        print(traceback.format_exc())
        raise e

# One time Method to be tried in ipython to find mean and std
def get_mean_std_channels(data_dir, crop_size, bs=4):
    try:
        data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        image_dataset = datasets.ImageFolder(data_dir, data_transforms)
        dataloader = DataLoader(image_dataset, batch_size=bs, 
                        shuffle=True, num_workers=bs) 
        pop_mean = []
        pop_std = []
        for i, data in enumerate(dataloader, 0):
            np_img = data[0].numpy()
            batch_mean = np.mean(np_img, axis=(0, 2, 3))
            batch_std = np.std(np_img, axis=(0, 2, 3))

            pop_mean.append(batch_mean)
            pop_std.append(batch_std)

        pop_mean = np.array(pop_mean).mean(axis=0)
        pop_std = np.array(pop_std).mean(axis=0)
        return pop_mean, pop_std
    except Exception as e:
        print(traceback.format_exc())
        raise e
