# Fixed Classification Layer in Image Classification models

Gabi Shalev (shalev.gabi@gmail.com)<br/>
Gal Lev Shalev (gallev898@gmail.com)<br/>
Joseph Keshet (jkeshet@cs.biu.ac.il)<br/>

This repository provides a PyTorch implementation of the paper [Redesigning the Classification Layer by Randomizing the Class Representation Vectors](https://arxiv.org/abs/2011.08704).

<br/><br/>

![Model scheme](https://github.com/MLSpeech/FixedClassificationLayer/blob/main/images/fixed_layer.png)


<br/>

The repository allows training fixed and non-fixed dot-product models, and also fixed and non-fixed cosine-similarity maximization models on STL dataset (more datasets will be supported soon). ResNet-18 is set as the visual encoder component.
