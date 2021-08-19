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


def create_UNET():
    """
    Returns a standard UNET sequential model, implemented in Keras, constructed
    according to the reference diagram (see next cell)

    Notes:
    Inputs are 128 * 128 RGB images (in the format of numpy arrays)
    Outputs are 128 * 128 binary arrays, with 0 signifying "Inside the cell" and 1 signifying "Outside the cell"
    """
    inputs = Input((128, 128, 3))

    s = Lambda(lambda x: x / 255.0)(inputs)  # normalise input to value between 0 and 1, and convert inputs to floats

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)  # Drop out 0.1 or 10% of the inputs from c1
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    # Kernel initialiser defines the initial values for the kernel (he_normal is a truncated gaussian / normal dist
    p1 = MaxPooling2D((2, 2))(
        c1)  # pools each 2x2 square of pixels by taking the max value. Downsizes the image by a factor of two

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.1)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.1)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.1)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate()([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.1)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate()([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.1)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1])
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return tf.keras.Model(inputs=[inputs], outputs=[outputs])


# Retrieved from https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def dice_coef(y_true, y_pred, smooth=1):
    """
    Computes the dice coefficient of the tensors y_true and y_pred.
    The dice coefficient is effectively a measure of the proportion of "overlap", where a higher score is preferable.
    This is used as a performance metric for the UNET model during training.

    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice_coefficient = (2. * intersection + smooth) / (
                K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

    return dice_coefficient


# retrieved from https://datascience.stackexchange.com/questions/58735/weighted-binary-cross-entropy-loss-keras-implementation
def weighted_bce(y_true, y_pred, mult=9):
    """
    Utilises the Keras built-in binary cross-entropy loss function to calculate a weighted version from input tensors.
    This is achieved by multiplying the ground truth mask tensor by a multiplier to increase (or decrease) their weight.

    The technique is useful in dealing with unbalanced datasets, where the ground truth images feature significantly
    more foreground than background (or vice versa).

    It is useful here as pixels that should be classified 'inside the cell' (foreground) far outnumber those that
    should be classified 'outside the cell' (background).
    Hence the weighting prevents the model from simply classifying all cells as the latter for a high accuracy score.
    """
    weights = (y_true * 9) + 1.
    bce = K.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)

    return weighted_bce


def ImportImages(raw_path, label_path, raw_suffix='.png', label_suffix='.png'):
    """
    Imports Raw and Label images from specified directory paths.
    It returns a list of image_ids (derived from the filenames) as well as two dictionaries
    (keys are ids, values are the corresponding raw or label images as numpy arrays).
    It is assumed that the raw image and label images share the same base filename in the directory, with an optional suffix for each.
    """

    print("Importing Images ...")
    print(len(raw_suffix))
    image_ids = [filename[:-len(raw_suffix)] for filename in
                 next(os.walk(raw_path))[2]]  # removes the suffix, generating ids
    raw_images, label_images = dict(), dict()

    for image_id in image_ids:
        rimage_path = os.path.join(raw_path, image_id + raw_suffix)
        limage_path = os.path.join(label_path, image_id + label_suffix)
        print(limage_path, raw_path)
        raw_image = Image.open(rimage_path)
        raw_image_arr = np.array(raw_image, dtype=np.float32)

        limage_path = ImageOps.grayscale(Image.open(limage_path))
        label_image_arr = np.array(limage_path, dtype=np.float32)
        label_image_arr = np.expand_dims(label_image_arr, -1)

        raw_images[image_id] = raw_image_arr
        label_images[image_id] = (label_image_arr > 0).astype(np.float32) * 255

    print('Images imported!')

    return image_ids, raw_images, label_images

def training_generator(raw_images, label_images,
                       threshold=255 / 2,
                       input_channels=3,
                       output_channels=1,
                       target_dimensions=(128, 128),
                       batch_size=32,
                       sigma=100,
                       seed=42):
    """
    This function creates a python generator object which creates batches of augmented training images 'on the fly'.
    Generating images during training is more efficient for memory (only enough for one batch is needed, as each batch
    is used once before being overwritten).

    The generator is initialised by:
    - Creating a Keras ImageDataGenerator object for both raw and label images with the same distortion parameters
    This ensures that the label and image will match following augmentation.
    - Creating Zero-filled numpy arrays of appropriate dimensions. These will receive the final augmented images

    Each image in the batch is generated by:
    - Random selection of a raw / label image pair.
    - Calculating appropriate random crop co-ordinates
    - Deform the corresponding crop of both the raw and label images (in the same way) using the elasticdeform library.
    This is a computationally expensive step.
    - Use the Keras ImageDataGenerator objects to further augment the images
    - Save the output as the final image and label in the ith entry in their respective numpy arrays.

    At the moment, this process is very slow. I think this is due to the fact that a new ImageDataGenerator object
    needs to be initialised for each random crop, when it was intended to be initialised once, and used many times.
    The elastic deformation slows things down too. However, the output is as expected. (See Example Training Images)

    Next steps would be to further optimise this code so images of comparable quality can be generated faster.
    """

    def get_training_crop_boundaries(image):
        # Generates co-ordinates to be used to crop a random subsection of the input image with the target dimensions
        nonlocal target_dimensions

        target_height, target_width = target_dimensions
        original_height, original_width, __ = image.shape

        left = random.randint(0, original_width - target_width)
        right = left + target_width

        top = random.randint(0, original_height - target_height)
        bottom = top + target_height

        return left, right, top, bottom

    image_data_generator_args = dict(
        rotation_range=90,
        height_shift_range=0.1,
        width_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.2,
        channel_shift_range=40.,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="reflect"
    )

    # Create a Keras ImageDataGenerator object for both raw and label images
    raw_image_gen = ImageDataGenerator(**image_data_generator_args)
    label_image_gen = ImageDataGenerator(**image_data_generator_args)

    # Initialise numpy arrays to receive generated images
    # Axis 0 is dictated by 'batch_size', while other axes are determined by the shape of the output image
    image_batch = np.zeros((batch_size, target_dimensions[0], target_dimensions[1], input_channels), dtype=np.float32)
    label_batch = np.zeros((batch_size, target_dimensions[0], target_dimensions[1], output_channels), dtype=np.float32)

    while True:
        for i in range(batch_size):
            # randomly select an image from the training set
            random_id = random.choice(list(raw_images.keys()))
            image_arr = raw_images[random_id]
            label_arr = label_images[random_id]

            # generate appropriate random crop boundaries
            left, right, top, bottom = get_training_crop_boundaries(image_arr)

            # perform random elastic deformation on the input arrays, specifying these crop co-ordinates
            # This is done using the elasticdeform library.
            # Only the smaller cropped region is computed (not the entire image), increasing efficiency
            [image_crop_deform_arr, label_crop_deform_arr] = elasticdeform.deform_random_grid([image_arr, label_arr],
                                                                                              axis=[(0, 1), (0, 1)],
                                                                                              sigma=sigma, crop=(
                    slice(top, bottom),
                    slice(left, right)
                )
                                                                                              )

            # Expand the dimensions of both the image and label arrays
            # ImageDataGenerator requires a 4D input.
            image_crop_deform_arr = np.expand_dims(image_crop_deform_arr, 0)
            label_crop_deform_arr = np.expand_dims(label_crop_deform_arr, 0)

            # Create an iterator for the cropped & deformed array from the ImageDataGenerator
            # objects created earlier by calling the '.flow()' method with the same random seed.
            # This will allow the same augmentation operations to be applied to both the raw image and the mask
            image_iter = raw_image_gen.flow(image_crop_deform_arr, seed=seed)
            label_iter = label_image_gen.flow(label_crop_deform_arr, seed=seed)

            # get the next image from the augmentation iterator
            final_label = next(label_iter)
            final_image = next(image_iter)

            # Assign the image and label to the output arrays
            image_batch[i] = final_image
            label_batch[i] = final_label > threshold

        yield image_batch, label_batch

def validation_generator(raw_images, label_images, threshold=255 / 2, input_channels=3, output_channels=1,
                         overlap_dec=0.3, target_dimensions=(128, 128), batch_size=32, seed=42):
    """
    This function creates a python generator object which creates batches of validation images 'on the fly'
    (no augmentation, just cropping).

    Images are not augmented in any way, only cropped. At a high level, images are obtained by 'moving a crop window' over
    the validation data, with an overlap proportion (specified by overlap_dec).

    Generating images during training is more efficient for memory (only enough for one batch is needed, as each batch
    is used once before being overwritten).

    The generator is initialised by:
    - Creating 'counters' to keep track of the 'current crop' so as to avoid repetitions.
    - Creating Zero-filled numpy arrays of appropriate dimensions. These will receive the data for each batch of images

    Each image in the batch is generated by:
    - Selection of the current validation image
    - Calculating appropriate crop co-ordinates using the 'counter variables', updating these variables appropriately.
    - Save the output as the final image and label in the ith entry in their respective numpy arrays.


    Next steps would be to investigate ways of getting different images in subsequent iterations: currently,
    when the bottom-right corner crop of the final validation image is captured, the counters return to the top-left corner
    of the first image. This means the validation images effectively loop.

    The benefit of this approach is that every part of every validation input image will be featured in the validation set.
    """

    # initialise variables to track the crop position in the image. These will be updated as validation images are generated.
    curr_row, curr_col = 0, 0
    source_index = 0
    final_crop_of_image = False

    # Initialise numpy arrays to receive generated images
    # Axis 0 is dictated by 'batch_size', while other axes are determined by the shape of the output image
    image_batch = np.zeros((batch_size, target_dimensions[0], target_dimensions[1], input_channels), dtype=np.float32)
    label_batch = np.zeros((batch_size, target_dimensions[0], target_dimensions[1], output_channels), dtype=np.float32)

    def get_val_crop_boundaries(image):
        # Generates co-ordinates to be used to crop a subsection of the input image with the target dimensions
        # These crop co-ordinates progressively move through a grid, with an overlap of 'overlap_dec*100'%

        nonlocal curr_row, curr_col, source_index, final_crop_of_image

        target_height, target_width = target_dimensions
        original_height, original_width, __ = image.shape

        left = 0 + curr_col * (1 - overlap_dec) * target_width
        right = left + target_width
        curr_col += 1

        top = 0 + curr_row * (1 - overlap_dec) * target_height
        bottom = top + target_height

        if bottom > original_height and right > original_width:
            final_crop_of_image = True

        if right > original_width:
            # If final image of the row, then take rightmost patch of 'target_size'
            right = original_width
            left = right - target_width
            # reset to the left-hand column
            curr_col = 0
            # move to the next row
            curr_row += 1

        if bottom > original_height:
            # If in the final row, then take the bottommost patch of 'target_size'
            bottom = original_height
            top = bottom - target_height

        if final_crop_of_image:
            # if in the bottom-right corner of the image,
            # move on to the next validation image.
            # If currently on the last image, return to the first one.
            source_index = (source_index + 1) % len(raw_images)
            final_crop_of_image = False
            # reset counters to top-left corner
            curr_row = 0
            curr_col = 0

        return int(left), int(right), int(top), int(bottom)

    while True:
        for i in range(batch_size):
            # Retrieve the image and label of the current validation source image
            image_id = list(raw_images.keys())[source_index]
            image_arr = raw_images[image_id]
            label_arr = label_images[image_id]

            # Compute validation boundaries
            left, right, top, bottom = get_val_crop_boundaries(image_arr)

            # Assign final image to the batch arrays
            image_batch[i] = image_arr[top:bottom, left:right, :]
            label_batch[i] = label_arr[top:bottom, left:right] > threshold

        yield image_batch, label_batch