import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets.aptos2019 import APTOS2019Dataset
from utils.transforms import CricleCrop, Normalize
from models.blindness_detection import BlindnessDetection
from trainers.blindness_trainer import BlindnessDetectionTrainer

if __name__ == "__main__":

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  transform = transforms.Compose([
      CricleCrop(device=device),
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomVerticalFlip(),
      transforms.RandomRotation((0, 15)),
      Normalize(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std =[0.229, 0.224, 0.225])
  ])

  test_transform = transforms.Compose([
      CricleCrop(),
      transforms.Resize((224, 224)),
      Normalize(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std =[0.229, 0.224, 0.225])
  ])

  train_data = APTOS2019Dataset(root='data', dataset_split='train', download=True, transform=transform)
  val_data = APTOS2019Dataset(root='data', dataset_split='val', download=True, transform=test_transform)

  train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
  val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False)

  weights = models.VGG19_BN_Weights.DEFAULT
  extractor = models.vgg19_bn(weights=weights)

  model = BlindnessDetection(extractor, 128, 5)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), betas=(0.85, 0.95), weight_decay=0.0001)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
  mixed_precision = False

  trainer = BlindnessDetectionTrainer(model, criterion, optimizer, scheduler, device, mixed_precision)
  trainer.fit(10, train_dataloader, val_dataloader, 2)
