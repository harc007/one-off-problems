import cv_utils as cv
import os

show_image = True
ccmean = [0.5438018, 0.5438018, 0.5438018]
ccstd = [0.18383098, 0.18383098, 0.18383098]
crop_size = 300
data_dir = os.path.join(os.getcwd(), 'chest_xray')
weights = [0.5, 0.5]
num_samples = 2000
mname = 'resnet18'
pretrained = True
lossname = 'crossentropy'
optimname = 'sgd'
is_fc = True
lr = 0.001
momentum = 0.9
schname = 'step'
step_size = 7
gamma = 0.1
num_epochs = 10
num_ftrs = 8192


dtransform = cv.get_data_transforms(ccmean, ccstd,
                crop_size)

dloaders, dsizes, all_classes, device = cv.get_dataloader_obj(data_dir, 
                                            dtransform, weights, num_samples)

inputs, classes = next(iter(dloaders['train']))


if show_image:
    titles = [all_classes[x] for x in classes]
    cv.imshow(inputs, ccmean, ccstd, title=titles)

model = cv.get_model(mname, len(all_classes), device, num_ftrs, pretrained=pretrained)

criterion = cv.get_loss_criteria(lossname)

optimizer = cv.get_optimizer(optimname, is_fc, lr, momentum, model)

scheduler = cv.get_scheduler(schname, optimizer, step_size, gamma)

model_ft = cv.train_model(model, criterion, optimizer, scheduler, num_epochs, dloaders, device, dsizes)
