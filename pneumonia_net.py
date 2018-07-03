import cv_utils as cv
import os

show_image = True
ccmean = [0.5438018, 0.5438018, 0.5438018]
ccstd = [0.18383098, 0.18383098, 0.18383098]
crop_size = 800
data_dir = os.path.join(os.getcwd(), 'chest_xray')
mname = 'resnet18'
pretrained = True


dtransform = cv.get_data_transforms(ccmean, ccstd,
                crop_size)

dloaders, dsizes, all_classes, device = cv.get_dataloader_obj(data_dir, dtransform)

inputs, classes = next(iter(dloaders['train']))


if show_image:
    titles = [all_classes[x] for x in classes]
    cv.imshow(inputs, ccmean, ccstd, title=titles)

model = cv.get_model(mname, len(all_classes), device, pretrained=pretrained)
print(model)
