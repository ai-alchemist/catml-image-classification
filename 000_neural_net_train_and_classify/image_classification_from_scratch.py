"""
This file shows how to train an image classification model, and perform inference with Keras.

This contains various functionality and quality-of-life features, like support
for more than two classes, command-line arguments, and sanity checks.
"""

"""
## Introduction

This example shows how to do image classification from scratch, starting
from image image files on disk, without leveraging pre-trained weights
or a pre-made Keras Application model.

We use the `image_dataset_from_directory` utility to generate the
datasets, and we use Keras image preprocessing layers for image
standardization and data augmentation.
"""

"""
## Setup
"""

import os
import argparse

import numpy as np
import keras
from keras import layers, ops
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

"""
## Load the dataset.
"""

"""shell
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
unzip -q kagglecatsanddogs_5340.zip
ls
# Now there is a `PetImages` folder which contain two subfolders, `Cat` and `Dog`.
# Each subfolder contains image files for each category.
"""

"""
Alternatively, the following dataset is good for demonstrating classification
of more than two classes:

https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

This can be downloaded directly from:

https://www.kaggle.com/api/v1/datasets/download/samithsachidanandan/human-face-emotions
"""

"""
### Filter out corrupted images

When working with some datasets, corrupted images are sometimes present.
Optionally filter out badly-encoded images that do not feature the
string "JFIF" in their header.
"""

########################################

####################
# Parsing Arguments
####################

parser = argparse.ArgumentParser()
parser.add_argument('data_path',
                    help='Path to dataset directory.')
parser.add_argument('--model_dir', default='model_files',
                    help='Directory to save the trained model.')
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--validation_split', type=float, default=0.2,
                    help='Portion of data to use for validation.')
parser.add_argument('--demo_image', default=None,
                    help='Demonstration image to perform inference on.')
parser.add_argument('--delete_corrupted_files', action='store_true',
                    help='Automatically delete corrupted files from the dataset')
args = parser.parse_args()

# assign local variables values from cmdline args.
data_path = args.data_path
model_dir = args.model_dir
num_epochs = args.epochs
batch_size = args.batch_size
validation_split = args.validation_split

delete_corrupted_files = args.delete_corrupted_files

demo_image_path = args.demo_image

# print the arguments.
print("####ARGUMENTS####")
print("Data Path: " + data_path)
print("Model Directory: " + model_dir)
print("Number of Epochs: " + str(num_epochs))
print("Batch size: " + str(batch_size))
print("Validation Split: " + str(validation_split))
print("Demo Image Path: " + demo_image_path)

# Create the chosen model directory, if necessary.
os.makedirs(args.model_dir, exist_ok=True)

########################################

if (delete_corrupted_files):
    # Try to remove corrupted images.
    num_skipped = 0
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = b"JFIF" in fobj.peek(10)
            finally:
                fobj.close()
    
            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
    
    print(f"Deleted {num_skipped} images.")

########################################

"""
## Generate a `Dataset`
"""

image_size = (180, 180)

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_path,
    validation_split=validation_split,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="categorical",
)

# The number of output neurons is determined by the number of
# classes in the dataset.
num_classes = len(train_ds.class_names)
print("Neural net will be compiled for " + str(num_classes) + " classes.")
print("The classes are: " + str(train_ds.class_names))

"""
## Visualize the data

Here are the first 9 images in the training dataset.
"""

# This is good as a sanity check, but it's a blocking operation,
# so it's disabled for normal unattended runs.

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
    # for i in range(9):
        # ax = plt.subplot(3, 3, i + 1)
        # plt.imshow(np.array(images[i]).astype("uint8"))
        # plt.title(int(labels[i]))
        # plt.axis("off")
        # plt.show()

"""
## Using image data augmentation

When you don't have a large image dataset, it's a good practice to artificially
introduce sample diversity by applying random yet realistic transformations to the
training images, such as random horizontal flipping or small random rotations. This
helps expose the model to different aspects of the training data while slowing down
overfitting.
"""

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]


def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images


"""
Let's visualize what the augmented samples look like, by applying `data_augmentation`
repeatedly to the first few images in the dataset:
"""

# This is good as a sanity check, but it's a blocking operation,
# so it's disabled for normal unattended runs.

# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
    # for i in range(9):
        # augmented_images = data_augmentation(images)
        # ax = plt.subplot(3, 3, i + 1)
        # plt.imshow(np.array(augmented_images[0]).astype("uint8"))
        # plt.axis("off")
        # plt.show()


"""
## Standardizing the data

Our image are already in a standard size (180x180), as they are being
yielded as contiguous `float32` batches by our dataset.

However, their RGB channel values are in the `[0, 255]` range.
This is not ideal for a neural network; in general you should seek to
make your input values small. 

Here, we will standardize values to be in the `[0, 1]` by using a
`Rescaling` layer at the start of our model.
"""

"""
## Two options to preprocess the data

There are two ways you could be using the `data_augmentation` preprocessor:

**Option 1: Make it part of the model**, like this:

```python
inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
...  # Rest of the model
```

With this option, your data augmentation will happen *on device*, synchronously
with the rest of the model execution, meaning that it will benefit from GPU
acceleration.

Note that data augmentation is inactive at test time, so the input samples will only be
augmented during `fit()`, not when calling `evaluate()` or `predict()`.

If you're training on GPU, this may be a good option.

**Option 2: apply it to the dataset**, so as to obtain a dataset that yields batches of
augmented images, like this:

```python
augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
```

With this option, your data augmentation will happen **on CPU**, asynchronously, and will
be buffered before going into the model.

If you're training on CPU, this is the better option, since it makes data augmentation
asynchronous and non-blocking.

In our case, we'll go with the second option. If you're not sure
which one to pick, this second option (asynchronous preprocessing) is always a solid choice.
"""

"""
## Configure the dataset for performance

Let's apply data augmentation to our training dataset,
and let's make sure to use buffered prefetching so we can yield data from disk without
having I/O becoming blocking:
"""

# Apply `data_augmentation` to the training images.
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf_data.AUTOTUNE,
)
# Prefetching samples in GPU memory helps maximize GPU utilization.
train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
val_ds = val_ds.prefetch(tf_data.AUTOTUNE)

"""
## Build a model

We'll build a small version of the Xception network.
If you want to do a systematic search for the best model
configuration, consider using
[KerasTuner](https://github.com/keras-team/keras-tuner).

Note that:

- We start the model with the `data_augmentation` preprocessor, followed by a
 `Rescaling` layer.
- We include a `Dropout` layer before the final classification layer.
"""


def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    # Determine the number of output neurons to use.
    # Each class needs one output neuron.
    units = num_classes

    x = layers.Dropout(0.25)(x)
    
    # Use "softmax" activation function to constrain outputs to
    # between 0 and 1.
    outputs = layers.Dense(units, activation='softmax')(x)
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=num_classes)
keras.utils.plot_model(model, show_shapes=True)

"""
## Train the model
"""

epochs = num_epochs

callbacks = [
    keras.callbacks.ModelCheckpoint(model_dir+"/"+"save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(3e-4),
    loss=keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

####################

# Save the model.
model.save(model_dir+"/"+'neural_net.keras')

# Load the model from the file. This is mainly for verification.
model = keras.models.load_model(model_dir+"/"+'neural_net.keras')

####################

"""
We get to >90% validation accuracy after training for 25 epochs on the full dataset
(in practice, you can train for 50+ epochs before validation performance starts degrading).
"""

"""
## Run inference on new data

Note that data augmentation and dropout are inactive at inference time.
"""

if(demo_image_path):

    img = keras.utils.load_img(demo_image_path, target_size=image_size)
    plt.imshow(img)
    
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis.
    
    # Predictions with support for multiple class types.
    predictions = model.predict(img_array)
    
    # Print predictions in decimal, not scientific notation.
    np.set_printoptions(suppress=True)
    
    # This prints the confidence that the image contains each class.
    # It is up to the user to interpret the values as the relevant
    # object types.
    print("Predictions for " + demo_image_path + ": " + str(predictions))



