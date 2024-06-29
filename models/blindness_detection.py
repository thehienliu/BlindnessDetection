from torch import nn

class BlindnessDetection(nn.Module):
  def __init__(self, extractor, hidden_size, output_size):
    super().__init__()

    self.extractor = extractor.features
    self.avgpool = extractor.avgpool
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(extractor.classifier[0].in_features, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, output_size)
    )

  def forward(self, x):
    x = self.extractor(x)
    x = self.avgpool(x)
    return self.classifier(x)
