from torch import nn
from torchvision import models


class ResNet50:
    def __init__(self) -> None:

        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.avgpool = nn.Sequential(model.avgpool, nn.Flatten())
        self.input_size = 224
        self.output_size = 2048


class VGG19:
    def __init__(self) -> None:

        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)
        self.features = model.features
        self.avgpool = model.avgpool
        self.input_size = 224
        self.output_size = 25088


class EfficientNetV2:
    def __init__(self) -> None:
        model = models.efficientnet_v2_l(
            weights=models.EfficientNet_V2_L_Weights.DEFAULT
        )
        self.features = model.features
        self.avgpool = model.avgpool
        self.input_size = 480
        self.output_size = 1280


class SwinTransformer:
    def __init__(self) -> None:
        model = models.swin_v2_b(models.Swin_V2_B_Weights.DEFAULT)
        self.features = model.features
        self.avgpool = nn.Sequential(
            model.norm, model.permute, model.avgpool, model.flatten
        )
        self.input_size = 256
        self.output_size = 1024


def retrieve_extractor_fn(module_name: str):

    if not module_name in EXTRACTOR_MODULE:
        raise ValueError(
            f"Expect module_name in {list(EXTRACTOR_MODULE.keys())}, got {module_name}!"
        )

    extractor = EXTRACTOR_MODULE[module_name]
    return extractor


EXTRACTOR_MODULE = {
    "resnet50": ResNet50,
    "vgg19": VGG19,
    "efficientnetv2": EfficientNetV2,
    "swintransformer": SwinTransformer,
}
