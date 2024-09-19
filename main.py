import os
import sys
import torch
import loguru
import argparse
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from datetime import date
from torch.utils.data import DataLoader
from torchvision import transforms, models
from datasets.aptos2019 import APTOS2019Dataset
from utils.transforms import CricleCrop, Normalize
from models.blindness_detection import BlindnessDetection
from trainers.blindness_trainer import BlindnessDetectionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Blindness Detection Model")
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration's path that contains transform, model, training information.",
    )

    parser.add_argument(
        "--output",
        default=f"{str(date.today())}",
        type=str,
        help="Output directory for model and log file.",
    )

    parser.add_argument(
        "--output_log_name",
        default="file.log",
        type=str,
        help="Output name for log file.",
    )

    parser.add_argument(
        "--output_model_name",
        default="model.pt",
        type=str,
        help="Output name for model state_dict.",
    )

    return parser.parse_args()


def instantiate_logger(output_log: str = "file.log") -> loguru.logger:

    log_level = "DEBUG"
    logger.remove()
    logger.add(
        sys.stderr, level=log_level, colorize=True, backtrace=True, diagnose=True
    )
    logger.add(
        output_log, level=log_level, colorize=False, backtrace=True, diagnose=True
    )
    return logger


def get_transform(config):

    train_transform = transforms.Compose(
        [
            CricleCrop(device=config.training.device),
            transforms.Resize(
                (config.transform.input_size, config.transform.input_size)
            ),
            transforms.RandomHorizontalFlip(p=config.transform.horizontal_flip),
            transforms.RandomVerticalFlip(p=config.transform.vertical_flip),
            transforms.RandomRotation(degrees=config.transform.rotation),
            Normalize(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            CricleCrop(),
            transforms.Resize(
                (config.transform.input_size, config.transform.input_size)
            ),
            Normalize(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


if __name__ == "__main__":

    # Get parser
    args = parse_args()
    config = OmegaConf.load(args.config)

    # Create output dir
    os.makedirs(name=f"{args.output}", exist_ok=True)

    # Get logger
    logger = instantiate_logger(os.path.join(args.output, args.output_log_name))
    logger.info("Instantiated Logger.")

    # Setup data
    train_transform, test_transform = get_transform(config)

    logger.info("Get dataset via Kaggle's beta API: ")
    train_data = APTOS2019Dataset(
        root="data", dataset_split="train", download=True, transform=train_transform
    )
    val_data = APTOS2019Dataset(
        root="data", dataset_split="val", download=True, transform=test_transform
    )
    test_data = APTOS2019Dataset(
        root="data", dataset_split="test", download=True, transform=test_transform
    )

    train_dataloader = DataLoader(
        train_data, batch_size=config.training.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config.training.batch_size, shuffle=False
    )

    # Setup model
    model = BlindnessDetection(
        module_name=config.model.extractor_module_name,
        hidden_size=config.model.hidden_size,
        output_size=config.model.output_size,
    )
    logger.info(f"Model: {model}")

    # Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.85, 0.95),
        weight_decay=0.0001,
        lr=config.training.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    mixed_precision = config.training.mixed_precision
    device = config.training.device

    logger.info(
        f"Load criterion: {criterion}\n"
        f"Load optimizer: {optimizer}\n"
        f"Load scheduler: {scheduler}\n"
        f"Using device: {device}"
    )

    logger.info("Instantiate Trainer")
    trainer = BlindnessDetectionTrainer(
        model, criterion, optimizer, scheduler, logger, device, mixed_precision
    )

    logger.info("Calling Trainer Fit")
    logger.info(f"Starting training, total number of epochs: {config.training.epochs}")
    trainer.fit(
        config.training.epochs,
        train_dataloader,
        val_dataloader,
        config.training.eval_every,
    )

    # Save model
    model_path = os.path.join(args.output, args.output_model_name)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Save model at {model_path}!")
