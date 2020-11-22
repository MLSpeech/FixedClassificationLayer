# Fixed Classification Layer in Image Classification models

This repository provides a PyTorch implementation of the paper [Redesigning the Classification Layer by Randomizing the Class Representation Vectors](https://arxiv.org/abs/2011.08704).

![Model scheme](https://github.com/MLSpeech/FixedClassificationLayer/blob/main/images/fixed_layer.png)


This repository allows training fixed and non-fixed dot-product models, and also fixed and non-fixed cosine-similarity maximization models on STL dataset (more datasets will be supported soon). The repository sets ResNet-18 as the visual encoder component.
