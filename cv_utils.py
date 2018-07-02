import numpy as np
import traceback
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


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
                            ]) for k in keys}
        return data_transforms
    except Exception as e:
        print(traceback.format_exc())
        raise e


def get_dataloader_obj(data_dir, data_transforms, 
                        data_types=['train', 'test', 'val'], bs=4):
    try:
        image_datasets = {x:datasets.ImageFolder(data_dir, data_transforms[x]) 
                            for x in data_types}
        dataloaders = {x:DataLoader(image_datasets[x], batch_size=bs,
                            shuffle=True, num_worker=bs) for x in data_types}
        return dataloader
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
