import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
data_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

data_transforms_resnet = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

data_transforms_resnet_plus = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p = 0.5), 
        transforms.RandomRotation(degrees = 15),
        transforms.Resize(size=[232], interpolation=InterpolationMode.BILINEAR),    # provided by doc
        transforms.CenterCrop(224), # provided by doc
        transforms.ToTensor(),  # provided by doc
        transforms.RandomErasing(p = 0.4, scale = (0.02, 0.3), ratio = (0.3, 3.3), value = 0, inplace = False), # Has to be after ToTensor, value 0 for black and white
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # provided by doc
    ]
)

data_transforms_wide_resnet = data_transforms_wide_resnet = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees = 15),
        transforms.Resize(256), # provided by doc
        transforms.CenterCrop(224), # provided by doc
        transforms.ToTensor(),  # provided by doc
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # provided by doc
    ]
)