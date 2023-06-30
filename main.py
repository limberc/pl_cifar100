import timm.models as models
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import LightningCLI
from torch.utils.data import Dataset
from torchmetrics import Accuracy
from torchvision.datasets import CIFAR10, CIFAR100
from typing import Optional


def get_dataset(data_path, dataset):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = CIFAR10(data_path, train=True, transform=train_transform, download=True)
        test_data = CIFAR10(data_path, train=False, transform=test_transform, download=True)
    elif dataset == 'cifar100':
        train_data = CIFAR100(data_path, train=True, transform=train_transform, download=True)
        test_data = CIFAR100(data_path, train=False, transform=test_transform, download=True)
    return train_data, test_data


class CIFARLightningModel(LightningModule):
    # pull out resnet names from torchvision models
    MODEL_NAMES = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
    )

    def __init__(
            self,
            arch: str = 'resnet34',
            lr: float = 0.4,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,
            data_path: str = './data',
            dataset: str = 'cifar100',
            batch_size: int = 512,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        # self.pretrained = pretrained
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.dataset = dataset
        self.batch_size = batch_size
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100
        else:
            raise ValueError("NOT SUPPORT DATASET.")
        self.model = models.__dict__[self.arch](num_classes=num_classes)
        self.train_datset, self.test_dataset = get_dataset(data_path, dataset)
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.train_acc1 = Accuracy(top_k=1)
        self.train_acc5 = Accuracy(top_k=5)
        self.eval_acc1 = Accuracy(top_k=1)
        self.eval_acc5 = Accuracy(top_k=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        # update metrics
        self.train_acc1(output, target)
        self.train_acc5(output, target)
        self.log("train_acc1", self.train_acc1, prog_bar=True)
        self.log("train_acc5", self.train_acc5, prog_bar=True)
        return loss_val

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        self.eval_acc1(output, target)
        self.eval_acc5(output, target)
        self.log("val_acc1", self.eval_acc1, prog_bar=True)
        self.log("val_acc5", self.eval_acc5, prog_bar=True)
        return loss_val

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True
        )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_datset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=27,
            pin_memory=True,
            drop_last=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=12,
            pin_memory=True,
        )
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            pin_memory=True,
        )
        return test_loader

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)


class ClassificationCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults(
            {
                'trainer.max_epochs': 200,
                'trainer.deterministic': True,
                'trainer.strategy': 'ddp',
                'trainer.gpus': 1
            }
        )


if __name__ == '__main__':
    cli = ClassificationCLI(CIFARLightningModel)
