import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import elasticdeform  # https://pypi.org/project/elasticdeform/
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Concatenate, Conv2DTranspose

import func1 as f1

# "PATH TO FOLDER CONTAINING 'raw' and 'label' image folders"
ROOT_PATH = 'C:\Assignments\Compsci 789'
RAW_PATH = os.path.join(ROOT_PATH, "raw")
LABEL_PATH = os.path.join(ROOT_PATH, "label")

IMG_WIDTH = IMG_HEIGHT = 128  # fixed size required for fully connected layer.
TARGET_DIMENSIONS = (IMG_WIDTH, IMG_HEIGHT)
INPUT_CHANNELS = 3  # RGB channels of each input image
OUTPUT_CHANNELS = 1  # Grayscale channel of each output image
OVERLAP_DEC = 0.3  # overlap decimal for validation images
TRAIN_BATCH_SIZE = 25
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE  # // 9 #number of validation images in each batch
NUM_VAL_IMAGES = 2

LABEL_THRESHOLD = 0.5  # Threshold for generating binary mask (for floats between 0 and 1)

IMAGE_IDS, RAW_IMAGES, LABEL_IMAGES = f1.ImportImages(RAW_PATH, LABEL_PATH, raw_suffix='.png', label_suffix='_mask.png')

"""
This is the architecture for future reporting purposes. 


NUM_VAL_IMAGES many (Two) validation images are randomly selected from the set of 10 available input images.
The rest are used as training images.
This selection occurs without replacement, so no subsequent modules will have the same validation image as any previous model.

Currently, I only do this once, to train a single model. The next step is to build & compare the performance of several models
each trained on a different subset of the available dataset. If all have comparable (favorable) performance, then 
we can conclude that the model is likely to be useful, and is not simply overfitting to any one partition.
"""

remaining_val_ids = IMAGE_IDS.copy()
random.shuffle(remaining_val_ids)

train_raw_images, val_raw_images, train_label_images, val_label_images = dict(), dict(), dict(), dict()

while remaining_val_ids:  # STRUCTURE FOR RESAMPLING
    for __ in range(NUM_VAL_IMAGES):
        val_ids = remaining_val_ids.pop()
    train_raw_images.clear()
    val_raw_images.clear()
    train_label_images.clear()
    val_label_images.clear()

    for key, value in RAW_IMAGES.items():
        if key in val_ids:
            val_raw_images[key] = value
        else:
            train_raw_images[key] = value

    for key, value in LABEL_IMAGES.items():
        if key in val_ids:
            val_label_images[key] = value
        else:
            train_label_images[key] = value

    train_gen = f1.training_generator(train_raw_images, train_label_images, batch_size=TRAIN_BATCH_SIZE)
    val_gen = f1.validation_generator(val_raw_images, val_label_images, batch_size=VAL_BATCH_SIZE)
    print('Data generators initialized...')

    gen_model = f1.create_UNET()
    print('Model created...')
    gen_model.compile(optimizer='adam', loss=f1.weighted_bce, metrics=['accuracy', f1.dice_coef])
    print('Model compiled...')
    print('Training start!')
    gen_model.fit(
        x=train_gen,
        epochs=5,
        steps_per_epoch=10,
        validation_data=val_gen,
        validation_steps=10,
        verbose=1,
        shuffle=True)
    print('Training Complete')
    break
