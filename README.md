# Benchmarks for M1 Max

We trained model over M1 Max and trained for **1 epoch** on image with (3, 224, 224) for 10,000 images.

We reported the total training time (s).

| Model          | #parms  | Batch Size | Time (M1 Max) | Time (3090ti) |
| -------------- | ------- | ---------- | ------------- | ------------- |
| ResNet50       | 23.7M   | 64         |               |               |
|                |         | 512        |               |               |
| ViT-Base       | 86.57M  | 64         |               |               |
|                |         | 512        |               |               |
| ViT-Large      | 304.20M | 64         |               |               |
|                |         | 512        |               |               |
| SwinT-Base     | 87.76M  | 64         |               |               |
|                |         | 512        |               |               |
| SwinT-Large    | 196.53M | 64         |               |               |
|                |         | 512        |               |               |
| ConvNeXt-Base  | 88.59M  | 64         |               |               |
|                |         | 512        |               |               |
| ConvNeXt-Large | 197.76M | 64         |               |               |
|                |         | 512        |               |               |

resnet50

vit_large_patch14_224

vit_base_patch16_224

swin_base_patch4_window7_224

swin_large_patch4_window7_224

convnext_base

convnext_large