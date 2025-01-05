"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_resnet, data_transforms_resnet_plus, data_transforms_wide_resnet  
from model import Net, ResNet50, ResNet50_plus, ResNet50_plus_v2, Wide_ResNet50_2, ResNet50_plus_default


class ModelFactory:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net()
        if self.model_name == "resnet50":
            return ResNet50()
        if self.model_name == "resnet50_plus":
            return ResNet50_plus(p_dropout=0.0)
        if self.model_name == "resnet50_plus_v2":
            return ResNet50_plus_v2(p_dropout=0.2)
        if self.model_name == "wide_resnet50_2":
            return Wide_ResNet50_2()
        if self.model_name == "resnet50_plus_default":
            return ResNet50_plus_default(p_dropout=0.0)
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        if self.model_name == "resnet50":
            return data_transforms_resnet
        if self.model_name == "resnet50_plus":
            return data_transforms_resnet_plus
        if self.model_name == "resnet50_plus_v2":
            return data_transforms_resnet_plus
        if self.model_name == "wide_resnet50_2":
            return data_transforms_wide_resnet
        if self.model_name == "resnet50_plus_default":
            return data_transforms_resnet_plus
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
