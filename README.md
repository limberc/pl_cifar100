# PyTorch Lightning CIFAR100

This is a extremely simple PyTorch Lighting repo for Computer Vision learner.

## Usage

You can first use `python main.py -h` to get the help.

You can easily change the basic settings in `default_config.yaml`. For example, the batch size and the arch (network).

## Training Details

I follow the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.4 divide by 5 at 60th, 120th, 160th epochs, train for 200 epochs. The batch size that we use is 512 with weight decay 5e-4, Nesterov momentum of 0.9.

To note that, for CIFAR10/100, the input shape is not as the same as the ImageNet. 
So, the model would slightly different than the ImageNet version which means the stride would be different.

Use ImageNet model directly would face up to the accuracy lost.

## Results

|   network   | params | VAL ACC@1 | VAL ACC@5 |
| :---------: | :----: | :-----: | :-----: |
|  ResNet18   | 11.2M  |  75.83  |  93.36  |
|  ResNet34   | 21.3M  |  77.53  |  94.11  |
|  ResNet50   | 23.7M  |  78.12  |  94.81  |
|  ResNet101  | 42.7M  |  78.53  |  95.13  |
|  ResNet152  | 58.3M  |  80.53  |  95.21  |
| MobileNetv2 |   2M   |  68.25  |  90.51  |
|  ResNeXt50  | 14.8M  |  79.54  |  95.21  |
| SeResNet18  | 11.4M  |  76.54  |  93.55  |
| SeResNet34  | 21.6M  |  76.13  |  93.14  |
| SeResNet50  | 26.5M  |  77.94  |  94.55  |
| DenseNet121 | 7.0M | 77.01 | 93.55 |
| DenseNet161 | 26M | 78.44 | 93.96 |