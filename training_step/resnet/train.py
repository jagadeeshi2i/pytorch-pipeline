import logging
import os
from random import sample

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader, IterableDataset
from torchvision import models, transforms

from model import CIFARCNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def train_model(
    train_glob: str,
    checkpoint_root: str
): 
  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
          
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    logger.log(logging.INFO, 'Creating model')
    model = CIFARCNN()

    logger.log(logging.INFO, 'Creating data loader')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)
        
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_root, 'weights')
    )

    logger.log(logging.INFO, 'Starting training')
    trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    max_epochs=1)

    trainer.fit(model, train_loader)
    torch.save(model, os.path.join(checkpoint_root, 'resnet18-5c106cde.pth'))

    return checkpoint_callback.best_model_path
  