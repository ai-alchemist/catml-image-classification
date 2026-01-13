# catml-image-classification

This repo contains working up-to-date programs for training neural nets and using them to classify images, as well as other related tasks.

These programs avoid hardcoding commonly-used options. Instead, they parse command-line arguments and provide sane defaults if necessary.

## 0. Neural Net Training and Classification

### Features

It shows how to create a `Dataset` object from a directory containing folders (which represent the classes) and images (which are the training data).

The training data is augmented by applying random transformations to the images.

The architecture of the neural net is explicitly laid out within the program.

The neural net is trained, the model is saved, and a demo image is optionally classified.

### How to Run

1. Install TensorFlow by running `python3 -m pip install tensorflow`. The exact command may be different for some systems.

2. Run `python3 image_classification_from_scratch.py path_to_dataset/`.

A full list of command-line arguments can be found by running  
`python3 image_classification_from_scratch.py -h`.

## 2. Multiclass Segmentation with DeepLabV3+

### Features

It shows how to create a `Dataset` object from a segmentation dataset, which tend to be more complex than classification datasets.

The architecture of the neural net is explicitly laid out within the program.

The neural net is trained, the model is saved, and demo images are segmented, showing the original image, the resulting masks, and a combination of both.

### How to Run

1. Install TensorFlow by running `python3 -m pip install tensorflow`. The exact command may be different for some systems.

2. Run `python3 deeplabv3_plus_segmentation.py path_to_dataset/`.

A full list of command-line arguments can be found by running  
`python3 deeplabv3_plus_segmentation.py -h`.