# Plant Classification with Lasagne/Theano

**Author** Ignacio Heredia

**Date** February 2017

This repository contains the code used to train a ResNet50 convolutional network on plant classification.
[Here](http://dl.acm.org/citation.cfm?doid=3075564.3075590) is the paper describing the work ([arxiv version](https://arxiv.org/abs/1706.03736)).

**Contents**

- `./data` Data files 
- `./scripts` Core code
- `./webpage` Independent folder which contains the essential files/functions to run a simple webpage to host your trained net and make predictions. For example the plant classification app is running at http://deep.ifca.es/.

This has been tested in Ubuntu 14.04 with Python 2.7.12 with the Anaconda 4.2.0 (64-bit) distribution, Theano 0.9.0.dev2 and Lasagne 0.2.dev1.

## Resusing this framework
This framework is quite flexible to retrain a ResNet50 with your image dataset (in `.jpg` format). 

### 1) Providing the dataset files

First you need add to the `./data/data_splits` path the files:

*Mandatory files*

- `synsets.txt`
- `train.txt`

*Optional files*

- `val.txt`
- `test.txt`
- `tags.txt`

The `train.txt`, `val.txt` and `test.txt` files associate an image to a label number (that has to *start at zero*). The `synsets.txt` file translates those label numbers to label names. Finally the `tags.txt` enables you to provide a tag to each training image to custom the data augmentation operations you apply to each image (see  the docstring of the `data_augmentation` function in the `./scripts/data_utils.py` file).

You can find examples of these files at  `./data/data_splits/dataset_demo`.

### 2) Downloading the pretrained model
You have to download the Lasagne Model Zoo pretrained weights with ImageNet from [here](https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/resnet50.pkl) and copy them to `./data/data_splits/pretrained_weights`.


### 3) Launching the training
Then you can launch the training executing `./scripts/train_runfile.py` once you have *updated the parameters* of the training inside the script (like the number of epochs, the folder path containing your images, etc). If you want to train with gpu you should create a `.theanorc` file in your `~` dir with a content similar to the following: 
```
[global]
device=gpu
floatX=float32
[cuda] 
root = /usr/local/cuda-8.0
[lib]
cnmem=.75
```
The weights of the trained net will be stored in `./scripts/training_weights` (in an `.npz` file) and the training information in `./scripts/training_info` (in a `.json` file). 

To learn how to use your freshly trained model for making predictions or plotting your training information, take a look at `./scripts/test_scripts/test_demo.py`. 
If you prefer to have a graphical interface, you can run a simple webpage to query your model. For more info check the [webpage docs](./webpage/webpage_docs.md).
### References

[1]: https://arxiv.org/abs/1612.07360
    

If you find this useful in your work please consider citing:
```
@inproceedings{Heredia2017,
  doi = {10.1145/3075564.3075590},
  url = {https://doi.org/10.1145/3075564.3075590},
  year  = {2017},
  publisher = {{ACM} Press},
  author = {Ignacio Heredia},
  title = {Large-Scale Plant Classification with Deep Neural Networks},
  booktitle = {Proceedings of the Computing Frontiers Conference on {ZZZ}  - {CF}{\textquotesingle}17}
}
```