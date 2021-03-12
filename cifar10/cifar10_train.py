import logging
import os
import shutil
from itertools import islice
from pathlib import Path
from random import sample

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import webdataset as wds
from PIL import Image
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from torch import nn
from torch.multiprocessing import Queue
from torch.utils.data import DataLoader, IterableDataset
from torchvision import models, transforms
import boto3
from botocore.exceptions import ClientError
import matplotlib.pyplot as plt

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        """
        Initialization of inherited lightning data module
        """
        super(CIFAR10DataModule, self).__init__()

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.valid_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                self.normalize,
            ]
        )

        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        self.args = kwargs

    def prepare_data(self):
        """
        Implementation of abstract class
        """

    @staticmethod
    def getNumFiles(input_path):
        return len(os.listdir(input_path)) - 1

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        data_path = self.args["train_glob"]

        train_base_url = data_path + "/train"
        val_base_url = data_path + "/val"
        test_base_url = data_path + "/test"

        train_count = self.getNumFiles(train_base_url)
        val_count = self.getNumFiles(val_base_url)
        test_count = self.getNumFiles(test_base_url)

        train_url = "{}/{}-{}".format(
            train_base_url, "train", "{0.." + str(train_count) + "}.tar"
        )
        valid_url = "{}/{}-{}".format(
            val_base_url, "val", "{0.." + str(val_count) + "}.tar"
        )
        test_url = "{}/{}-{}".format(
            test_base_url, "test", "{0.." + str(test_count) + "}.tar"
        )

        self.train_dataset = (
            wds.Dataset(train_url, handler=wds.warn_and_continue, length=40000 // 40)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm;jpg;jpeg;png", info="cls")
            .map_dict(image=self.train_transform)
            .to_tuple("image", "info")
            .batched(40)
        )

        self.valid_dataset = (
            wds.Dataset(valid_url, handler=wds.warn_and_continue, length=10000 // 20)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm", info="cls")
            .map_dict(image=self.valid_transform)
            .to_tuple("image", "info")
            .batched(20)
        )

        self.test_dataset = (
            wds.Dataset(test_url, handler=wds.warn_and_continue, length=10000 // 20)
            .shuffle(100)
            .decode("pil")
            .rename(image="ppm", info="cls")
            .map_dict(image=self.valid_transform)
            .to_tuple("image", "info")
            .batched(20)
        )

    def create_data_loader(self, dataset, batch_size, num_workers):
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        self.train_data_loader = self.create_data_loader(
            self.train_dataset,
            self.args["train_batch_size"],
            self.args["train_num_workers"],
        )
        return self.train_data_loader

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        self.val_data_loader = self.create_data_loader(
            self.valid_dataset,
            self.args["val_batch_size"],
            self.args["val_num_workers"],
        )
        return self.val_data_loader

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        self.test_data_loader = self.create_data_loader(
            self.test_dataset, self.args["val_batch_size"], self.args["val_num_workers"]
        )
        return self.test_data_loader


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network, optimizer and scheduler
        """
        super(CIFAR10Classifier, self).__init__()
        self.model_conv = models.resnet50(pretrained=True)
        for param in self.model_conv.parameters():
            param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        num_classes = 10
        self.model_conv.fc = nn.Linear(num_ftrs, num_classes)

        self.scheduler = None
        self.optimizer = None
        self.args = kwargs

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        out = self.model_conv(x)
        return out

    def training_step(self, train_batch, batch_idx):
        if batch_idx == 0:
            self.reference_image = (train_batch[0][0]).unsqueeze(0)
            #self.reference_image.resize((1,1,28,28))
            print("\n\nREFERENCE IMAGE!!!")
            print(self.reference_image.shape)
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc.compute())
        return {"loss": loss}

    def test_step(self, test_batch, batch_idx):

        x, y = test_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        if self.args["accelerator"] is not None:
            self.log("test_loss", loss, sync_dist=True)
        else:
            self.log("test_loss", loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc.compute())
        return {"test_acc": self.test_acc.compute()}

    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        if self.args["accelerator"] is not None:
            self.log("val_loss", loss, sync_dist=True)
        else:
            self.log("val_loss", loss)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc.compute())
        return {"val_step_loss": loss, "val_loss": loss}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]

    def makegrid(self, output, numrows):
        outer = torch.Tensor.cpu(output).detach()
        plt.figure(figsize=(20, 5))
        b = np.array([]).reshape(0, outer.shape[2])
        c = np.array([]).reshape(numrows * outer.shape[2], 0)
        i = 0
        j = 0
        while i < outer.shape[1]:
            img = outer[0][i]
            b = np.concatenate((img, b), axis=0)
            j += 1
            if j == numrows:
                c = np.concatenate((c, b), axis=1)
                b = np.array([]).reshape(0, outer.shape[2])
                j = 0

            i += 1
        return c

    def showActivations(self, x):

        # logging reference image
        self.logger.experiment.add_image(
            "input", torch.Tensor.cpu(x[0][0]), self.current_epoch, dataformats="HW"
        )

        # logging layer 1 activations
        out = self.model_conv.conv1(x)
        c = self.makegrid(out, 4)
        self.logger.experiment.add_image(
            "layer 1", c, self.current_epoch, dataformats="HW"
        )

    def training_epoch_end(self, outputs):
        self.showActivations(self.reference_image)

        # Logging graph
        if(self.current_epoch==0):
            sampleImg=torch.rand((1,3,64,64))
            self.logger.experiment.add_graph(CIFAR10Classifier(),sampleImg)


def train_model(
    train_glob: str,
    gpus: int,
    tensorboard_root: str,
    max_epochs: int,
    train_batch_size: int,
    val_batch_size: int,
    train_num_workers: int,
    val_num_workers: int,
    learning_rate: int,
    accelerator: str,
    model_save_path: str
):

    if accelerator == "None":
        accelerator = None
    if train_batch_size == "None":
        train_batch_size = None
    if val_batch_size == "None":
        val_batch_size = None

    dict_args = {
        "train_glob": train_glob,
        "max_epochs": max_epochs,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
        "train_num_workers": train_num_workers,
        "val_num_workers": val_num_workers,
        "lr": learning_rate,
        "accelerator": accelerator,
    }

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    dm = CIFAR10DataModule(**dict_args)
    dm.prepare_data()
    dm.setup(stage="fit")

    model = CIFAR10Classifier(**dict_args)
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=5, verbose=True
    )

    Path(model_save_path).mkdir(parents=True, exist_ok=True)

    if len(os.listdir(model_save_path)) > 0:
        for filename in os.listdir(model_save_path):
            filepath = os.path.join(model_save_path, filename)
            try:
                shutil.rmtree(filepath)
            except OSError:
                os.remove(filepath)

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_path,
        filename="cifar10_{epoch:02d}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    if os.path.exists(os.path.join(tensorboard_root, "cifar10_lightning_kubeflow")):
        shutil.rmtree(os.path.join(tensorboard_root, "cifar10_lightning_kubeflow"))

    Path(tensorboard_root).mkdir(parents=True, exist_ok=True)

    # Tensorboard root name of the logging directory
    tboard = TensorBoardLogger(tensorboard_root, "cifar10_lightning_kubeflow")
    lr_logger = LearningRateMonitor()

    trainer = pl.Trainer(
        gpus=gpus,
        logger=tboard,
        checkpoint_callback=checkpoint_callback,
        max_epochs=max_epochs,
        callbacks=[lr_logger, early_stopping],
        accelerator=accelerator,
    )

    trainer.fit(model, dm)
    trainer.test()


if __name__ == "__main__":

    import sys
    import json

    data_set = json.loads(sys.argv[1])[0]
    output_path = json.loads(sys.argv[2])[0]
    input_parameters = json.loads(sys.argv[3])[0]

    print("INPUT_PARAMETERS:::")
    print(input_parameters)

    tensorboard_root = input_parameters['tensorboard_root']
    max_epochs = input_parameters['max_epochs']
    train_batch_size = input_parameters['train_batch_size']
    val_batch_size = input_parameters['val_batch_size']
    train_num_workers = input_parameters['train_num_workers']
    val_num_workers = input_parameters['val_num_workers']
    learning_rate = input_parameters['learning_rate']
    accelerator = input_parameters['accelerator']
    gpus = input_parameters['gpus']


    train_model(
        train_glob=data_set,
        model_save_path=output_path,
        tensorboard_root=tensorboard_root,
        max_epochs=max_epochs,
        gpus=gpus,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        train_num_workers=train_num_workers,
        val_num_workers=val_num_workers,
        learning_rate=learning_rate,
        accelerator=accelerator
    )
