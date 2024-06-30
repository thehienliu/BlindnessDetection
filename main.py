import torch
import logging
import argparse
from torch import nn
from pytz import timezone
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets.aptos2019 import APTOS2019Dataset
from utils.transforms import CricleCrop, Normalize
from models.blindness_detection import BlindnessDetection
from trainers.blindness_trainer import BlindnessDetectionTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Blindness Detection Model")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for dataloader.")
    parser.add_argument('--hidden_size', type=int, default=128, help="Hidden size in model's classfier.")
    parser.add_argument('--output_size', type=int, default=5, help="Final output from model's classifier.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Start learning rate.")
    parser.add_argument('--epochs', type=int, default=20, help="Training epoch.")
    parser.add_argument('--mixed_precision', type=bool, default=False, help="Using float16 training.")

    return parser.parse_args()

def instantiate_logger() -> logging.Logger:

    # Setup logging
    timetz = lambda *args: datetime.now(timezone('Asia/Ho_Chi_Minh')).timetuple()
    logging.Formatter.converter = timetz

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        filename="basic.log",
        filemode="w"
    )
    logging.getLogger('PIL')
    logger = logging.getLogger()
    return logger

if __name__ == "__main__":

  # Get logger
  logger = instantiate_logger()
  logger.info("Instantiated Logger.")

  # Get parser
  args = parse_args()
  logger.info(f"Using device: {args.device}")


  transform = transforms.Compose([
      CricleCrop(device=args.device),
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

  logger.info("Get dataset via Kaggle's beta API: ")
  train_data = APTOS2019Dataset(root='data', dataset_split='train', download=True, transform=transform)
  val_data = APTOS2019Dataset(root='data', dataset_split='val', download=True, transform=test_transform)

  train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
  val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

  weights = models.VGG19_BN_Weights.DEFAULT
  extractor = models.vgg19_bn(weights=weights)
  model = BlindnessDetection(extractor=extractor, hidden_size=args.hidden_size, output_size=args.output_size)
  logger.info(f"Model: {model}")

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), betas=(0.85, 0.95), weight_decay=0.0001, lr=args.learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
  mixed_precision = args.mixed_precision
  logger.info("Load criterion: {criterion}")
  logger.info("Load optimizer: {optimizer}")
  logger.info("Load scheduler: {scheduler}")

  logger.info("Instantiate Trainer")
  trainer = BlindnessDetectionTrainer(model, criterion, optimizer, scheduler, logger, args.device, mixed_precision)

  logger.info("Calling Trainer Fit")
  logger.info(f"Starting training, total number of epochs: {args.epochs}")
  trainer.fit(args.epochs, train_dataloader, val_dataloader, 2)