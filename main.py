from typing import Optional

import timm.models as models
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningCLI
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData


class FakeDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 512):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        self.train_data = FakeData(10000, num_classes=1000, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
        self.test_data = FakeData(num_classes=1000, transform=transforms.Compose([
            transforms.ToTensor(),
        ]))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


class VisionLightningModel(LightningModule):
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
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.data_path = data_path
        self.dataset = dataset
        self.batch_size = batch_size
        self.model = models.__dict__[self.arch]()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        return loss_val

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = F.cross_entropy(output, target)
        return loss_val

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=True,
        )
        scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 120, 160], gamma=0.2
        )
        return [optimizer], [scheduler]

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)


if __name__ == '__main__':
    LightningCLI(
        VisionLightningModel, FakeDataModule,
        seed_everything_default=42,
        save_config_callback=None,
        trainer_defaults={
            "max_epochs": 1,
            "accelerator": "mps",
            "strategy": "auto",
            "profiler": "simple",
            "num_sanity_val_steps": 0,
            "devices": -1,
            "benchmark": True,
        }
    )
