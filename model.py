import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights, wide_resnet50_2, Wide_ResNet50_2_Weights

nclasses = 500


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model_resnet50.fc = nn.Linear(2048, nclasses)
        
    def forward(self, x):
        return self.model_resnet50(x)


class ResNet50_plus(nn.Module):
    def __init__(self, p_dropout = 0.0):    # p_dropout = 0.0, 0.1, 0.2, 0.5 were tested
        super(ResNet50_plus, self).__init__()
        self.model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model_resnet50.fc = nn.Sequential(
            nn.Dropout(p = p_dropout),
            nn.Linear(2048, nclasses)
        )
        
    def forward(self, x):
        return self.model_resnet50(x)
    
class ResNet50_plus_v2(nn.Module):
    def __init__(self, p_dropout = 0.2):    # p_dropout = 0.0, 0.2 were tested
        super(ResNet50_plus_v2, self).__init__()
        self.model_resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model_resnet50.fc = nn.Sequential(
            nn.Dropout(p = p_dropout),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, nclasses)
        )
        
    def forward(self, x):
        return self.model_resnet50(x)
    
class Wide_ResNet50_2(nn.Module):
    def __init__(self):
        super(Wide_ResNet50_2, self).__init__()
        self.model_wide_resnet50_2 = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.model_wide_resnet50_2.fc = nn.Sequential(
            nn.Dropout(p = 0.0),    # p_dropout = 0.0, 0.2 were tested
            nn.Linear(2048, nclasses)
        )
        
    def forward(self, x):
        return self.model_wide_resnet50_2(x)
    
class ResNet50_plus_default(nn.Module):
    def __init__(self, p_dropout = 0.0):
        super(ResNet50_plus_default, self).__init__()
        self.model_resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model_resnet50.fc = nn.Sequential(
            nn.Dropout(p = p_dropout),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, nclasses)
        )
        
    def forward(self, x):
        return self.model_resnet50(x)