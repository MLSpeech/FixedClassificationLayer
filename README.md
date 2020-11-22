# Fixed Classification Layer in Image Classification models

Gabi Shalev (shalev.gabi@gmail.com)<br/>
Gal Lev Shalev (gallev898@gmail.com)<br/>
Joseph Keshet (jkeshet@cs.biu.ac.il)<br/>

This repository provides a PyTorch implementation of the paper [Redesigning the Classification Layer by Randomizing the Class Representation Vectors](https://arxiv.org/abs/2011.08704).

<br/><br/>

![Model scheme](https://github.com/MLSpeech/FixedClassificationLayer/blob/main/images/fixed_layer.png)


<br/>

The repository allows training fixed and non-fixed dot-product models, and also fixed and non-fixed cosine-similarity maximization models on STL dataset (more datasets will be supported soon). ResNet-18 is set as the visual encoder component.

<br/>

### Installation instructions
* Python 3.6
* PyTorch 1.0.0 +


### Usage
##### 1. Cloning the repository
```bash
$ git clone https://github.com/MLSpeech/FixedClassificationLayer.git
$ cd FixedClassificationLayer
```

##### 2. Training

* For training NON-FIXED dot-product run the following command:
```
python run_stl.py --data_dir <PATH TO DATA> --save_dir <PATH TO SAVE MODELS> --runname <NAME OF RUN>
```

* For training FIXED dot-product run the following command:
```
python run_stl.py --data_dir <PATH TO DATA> --save_dir <PATH TO SAVE MODELS> --runname <NAME OF RUN> --fixed
```

* For training NON-FIXED cosine-similarity run the following command:
```
python run_stl.py --data_dir <PATH TO DATA> --save_dir <PATH TO SAVE MODELS> --runname <NAME OF RUN> --cosine --s <S VALUE>
```

* For training FIXED cosine-similarity run the following command:
```
python run_stl.py --data_dir <PATH TO DATA> --save_dir <PATH TO SAVE MODELS> --runname <NAME OF RUN> --cosine --s <S VALUE> --fixed
```


### Citation
If you find our work useful please cite:
```
@article{shalev2020redesigning,
  title={Redesigning the classification layer by randomizing the class representation vectors},
  author={Shalev, Gabi and Shalev, Gal-Lev and Keshet, Joseph},
  journal={arXiv preprint arXiv:2011.08704},
  year={2020}
}
```
