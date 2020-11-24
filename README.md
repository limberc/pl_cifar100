# PyTorch Lightning CIFAR100

`python main.py -a ${your_model} --gpus ${num_gpus} --distributed_backend ddp`

## Training Details

I follow the hyperparameter settings in paper [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552v2), which is init lr = 0.4 divide by 5 at 60th, 120th, 160th epochs, train for 200 epochs. The batch size that we use is 512 with weight decay 5e-4, Nesterov momentum of 0.9.

To note that, for CIFAR10/100, the input shape is not as the same as the ImageNet. 
So, the model would slightly different than the ImageNet version which means the stride would be different.

Use ImageNet model directly would face up to the accuracy lost.

## Results

|     network     | params | error@1 | error@5 |
| :-------------: | :----: | :-----: | :-----: |
|    ResNet18     | 11.2M  |  75.83  |  93.36  |
|    ResNet34     | 21.3M  |  77.53  |  94.11  |
|    ResNet50     | 23.7M  |         |         |
|    ResNet101    | 42.7M  |         |         |
|    ResNet152    | 58.3M  |         |         |
| PreactResNet18  | 11.3M  |         |         |
| PreactResNet34  | 21.5M  |         |         |
| PreactResNet50  | 23.9M  |         |         |
| PreactResNet101 | 42.9M  |         |         |
| PreactResNet152 | 58.6M  |         |         |
|    ResNeXt50    | 14.8M  |         |         |
|   ResNeXt101    | 25.3M  |         |         |
|   ResNeXt152    | 33.3M  |         |         |
|   DenseNet121   |  7.0M  |         |         |
|   DenseNet161   |  26M   |         |         |
|   DenseNet201   |  18M   |         |         |
|   SeResNet18    | 11.4M  |         |         |
|   SeResNet34    | 21.6M  |         |         |
|   SeResNet50    | 26.5M  |         |         |
|   SeResNet101   | 47.7M  |         |         |
|   SeResNet152   | 66.2M  |         |         |