import torch
import logging
from torch import nn
from typing import Union
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
from pytz import timezone
from datetime import datetime


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

class BlindnessDetectionTrainer:
  def __init__(self,
               model: nn.Module,
               criterion: nn.Module,
               optimizer: Optimizer,
               scheduler: _LRScheduler,
               device: Union[str, torch.device]='cpu',
               mixed_precision: bool=False):

    self.model = model
    self.device = device
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.mixed_precision = mixed_precision
    if self.mixed_precision:
      self.scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
      self.scaler = None

  def fit(self,
          epochs: int,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          eval_every: int = 1,
          continue_epoch: int = 0):
    """Fitting function to start training and validation of the trainer"""

    self.model.to(self.device)
    logger.debug(f"Send model to device: {next(self.model.parameters()).device}")

    for epoch in range(epochs):
      train_metrics = self.train_epoch(train_dataloader)
      logger.info(train_metrics)

      if (epoch + 1) % eval_every == 0:
        val_metrics = self.val_epoch(val_dataloader)
        logger.info(val_metrics)

      # Log learning rate
      curr_lr = self.optimizer.param_groups[0]['lr']

      if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
        self.scheduler.step(float(val_metrics["loss"]))
      else:
        self.scheduler.step()

      new_lr = self.optimizer.param_groups[0]['lr']
      logger.debug(f'Old lr: {curr_lr:.6f} - New lr: {new_lr:.6f}')

  def train_epoch(self, train_dataloader: DataLoader):
    """Training logic for a training epoch"""

    epoch_metrics = {'loss': [], 'accuracy': []}

    self.model.train()
    for batch in tqdm(train_dataloader):

        batch_metrics = self.step(batch, train=True)

        # Save metrics
        for k, v in batch_metrics.items():
          epoch_metrics[k].append(v)

    return epoch_metrics

  def eval_epoch(self, val_dataloader: DataLoader):
    """Evaluate logic for a val epoch"""

    epoch_metrics = {'loss': [], 'accuracy': []}

    self.model.eval()
    with torch.no_grad():
      for batch in tqdm(val_dataloader):

        batch_metrics = self.step(batch, train=False)

        # Save metrics
        for k, v in batch_metrics.items():
          epoch_metrics[k].append(v)

    return epoch_metrics

  def step(self, batch, train=True):

    if self.mixed_precision:
      with torch.autocast(device_type="cuda", dtype=torch.float16):

        # Forward pass
        outputs = self.model(batch['image'].to(self.device))
        loss = self.criterion(outputs, batch['label'].to(self.device))

        if train:
          # Backward pass
          self.optimizer.zero_grad()
          self.scaler.scale(loss).backward()
          self.scaler.step(self.optimizer)
          self.scaler.update()
          self.model.zero_grad()

    else:
      # Forward pass
        outputs = self.model(batch['image'].to(self.device))
        loss = self.criterion(outputs, batch['label'].to(self.device))

        if train:
        # Backward pass
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          self.model.zero_grad()

    batch_metrics = {}
    batch_metrics['loss'] = loss.item()
    batch_metrics['accuracy'] = (outputs.argmax(dim=-1) == batch['label'].to(self.device)).sum().item() / len(batch['label'])

    return batch_metrics
