import torch
import numpy as np
import traceback
from torchvision import transforms, datasets, utils, models
from torch.utils.data import DataLoader, sampler
from torch import device, cuda, optim
from torch.optim import lr_scheduler
import os
import matplotlib.pyplot as plt
import time
from torch import nn
import copy

def get_data_transforms(ccmean, ccstd, crop_size, 
                            data_types=['train', 'test', 'val']):
    try:
        if len(ccmean) != len(ccstd) or len(ccmean) != len(data_types):
            return None
        data_transforms = {k:transforms.Compose([
                                transforms.Resize((crop_size, crop_size)),
                                transforms.ToTensor(),
                                transforms.Normalize(ccmean, ccstd)
                            ]) for k in data_types}
        return data_transforms
    except Exception as e:
        print(traceback.format_exc())
        raise e


def get_dataloader_obj(data_dir, data_transforms, weights, num_samples,  
                        is_slr=False,data_types=['train', 'test', 'val'], bs=4):
    try:
        slr = sampler.WeightedRandomSampler(weights, num_samples)
        image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                            for x in data_types}
        dataloaders = {x:DataLoader(image_datasets[x], batch_size=bs, 
            shuffle=True, num_workers=bs) for x in data_types}
        if is_slr:
            dataloaders = {x:DataLoader(image_datasets[x], batch_size=bs, 
                sampler=slr, num_workers=bs) for x in data_types}
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


def get_model(mname, num_class, dev, num_ftrs, pretrained=True):
    try:
        model_ft = None
        if mname == 'resnet18' and pretrained:
            model_ft = models.resnet18(pretrained=True)
        elif mnane == 'resnet18' and not pretrained:
            model_ft = models.resnet18()
        
        if model_ft:
            model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs/10),
                                nn.Linear(num_ftrs/10, 2))
            model_ft = model_ft.to(dev)
            return model_ft
        return model_ft
    except Exception as e:
        print(traceback.format_exc())
        raise e


def get_loss_criteria(lossname):
    try:
        if lossname == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif lossname == 'nll':
            return nn.NLLLoss()
        else:
            return None
    except Exception as e:
        print(traceback.format_exc())
        raise e
   

def get_optimizer(optimname, is_fc, lr, momentum, model_ft):
    try:
        if optimname == 'sgd':
            if is_fc:
                return optim.SGD(model_ft.fc.parameters(), lr=lr, momentum=momentum)
            else:
                return optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)
        elif lossname == 'adadelta':
            if is_fc:
                return optim.Adadelta(model_ft.fc.parameters(), lr=lr)
            else:
                return optim.Adadelta(model_ft.parameters(), lr=lr)
        else:
            return None
    except Exception as e:
        print(traceback.format_exc())
        raise e


def get_scheduler(schname, optimizer, step_size, gamma):
    try:
        if schname == 'step':
            return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            return None
    except Exception as e:
        print(traceback.format_exc())
        raise e


def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, device, dataset_sizes):
    try:
        st = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                            phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - st
        print('Training complete in {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        return model
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
