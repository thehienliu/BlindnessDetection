from torch import nn
from torchvision import models
from .modules import retrieve_extractor_fn


class BlindnessDetection(nn.Module):
    def __init__(self, module_name: str, hidden_size: int, output_size: int):
        super().__init__()

        extractor = retrieve_extractor_fn(module_name)()
        self.extractor = extractor.features
        self.avgpool = extractor.avgpool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(extractor.output_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = self.extractor(x)
        x = self.avgpool(x)
        return self.classifier(x)
