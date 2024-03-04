import tensorflow as tf
from keras.src.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
import os
import random
from PIL import Image
import numpy as np
from skimage import filters
from sklearn.model_selection import train_test_split
from datetime import datetime

main_folder = 'preprocessed-images/'
GENUINE_SIGNATURES_PREFIX = 'c-'
FORGED_SIGNATURES_PREFIX = 'cf-'
NUM_INPUT_IMAGES = 10
RESULT_FOLDER = 'training_results/'

def create_cnn(input_shape, num_classes1, num_classes2):
    input_layer = layers.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(96, (11, 11), activation='relu', strides=4, padding='same')(input_layer)
    batch1 = tf.keras.layers.BatchNormalization()(conv1)

    pool1 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(batch1)

    conv2 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', strides=1, padding='same')(pool1)
    batch3 = tf.keras.layers.BatchNormalization()(conv2)

    pool2 = tf.keras.layers.MaxPooling2D((3, 3), strides=2)(batch3)

    conv3 = tf.keras.layers.Conv2D(384, (3, 3), activation='relu', strides=1, padding='same')(pool2)
    batch5 = tf.keras.layers.BatchNormalization()(conv3)

    conv4 = tf.keras.layers.Conv2D(384, (3, 3), activation='relu', strides=1, padding='same')(batch5)
    batch6 = tf.keras.layers.BatchNormalization()(conv4)

    conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', strides=1, padding='same')(batch6)
    batch7 = tf.keras.layers.BatchNormalization()(conv5)

    pool3 = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding='same')(batch7)

    flatten = tf.keras.layers.Flatten()(pool3)

    l2 = tf.keras.regularizers.L2(l2=1e-4)

    dense1 = layers.Dense(2048, activation='relu')(flatten)
    batch9 = tf.keras.layers.BatchNormalization()(dense1)
    dense2 = layers.Dense(2048, activation='relu')(batch9)
    batch10 = tf.keras.layers.BatchNormalization()(dense2)

    output1 = tf.keras.layers.Dense(num_classes1, activation='softmax', kernel_regularizer=l2, name='output1')(batch10)
    output2 = tf.keras.layers.Dense(num_classes2, activation='sigmoid', kernel_regularizer=l2, name='output2')(batch10)

    label_layer_1 = tf.keras.layers.Input((10,))
    label_layer_2 = tf.keras.layers.Input((1,))

    model = tf.keras.Model(inputs=[input_layer, label_layer_1, label_layer_2], outputs=[output1, output2])

    lambda_const = tf.constant(0.5)

    categorical_loss = tf.reduce_mean((1 - label_layer_2) * (1 - lambda_const) * (-tf.reduce_sum(label_layer_1 * tf.math.log(output1), axis=1)))

    binary_loss = lambda_const * tf.reduce_mean(-label_layer_2 * tf.math.log(output2) - (1 - label_layer_2) * tf.math.log(1 - output2))

    loss = categorical_loss + binary_loss

    model.add_loss(loss)

    return model


def load_images():
    inner_folders = [folder for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

    selected_folders = random.sample(inner_folders, NUM_INPUT_IMAGES)

    data_structure = []
    images = []
    y_true_1 = []
    y_true_2 = []

    for i, folder in enumerate(selected_folders, start=1):
        images_paths = [img for img in os.listdir(os.path.join(main_folder, folder)) if img.endswith(('.jpg', '.jpeg'))]

        for img_file in images_paths:
            img_path = os.path.join(main_folder, folder, img_file)
            image = Image.open(img_path)
            img_array = np.array(image)

            threshold_value = filters.threshold_otsu(img_array)
            binary_image = img_array > threshold_value

            #otsu_image = Image.fromarray(binary_image.astype('uint8') * 255)

            images.append(np.expand_dims(binary_image.astype(np.uint8), axis=-1))
            y_true_1.append(to_categorical(i-1, num_classes=NUM_INPUT_IMAGES))
            y_true_2.append(1 if img_file.startswith(FORGED_SIGNATURES_PREFIX) else 0)

            #data_structure.append({
            #    'image': otsu_image,
            #    'is_genuine': y_true_2[-1],
            #    'folder_number': y_true_1[-1]
            #})

    print('loaded images: '+str(len(images)))

    return np.array(images), np.array(y_true_1), np.array(y_true_2)#, data_structure

def combined_loss(y_true, y_pred, f_true, f_pred):
    """
    Calculates the combined loss of a categorical cross entropy and a binary cross entropy.

    Args:
    y_true: A vector containing the true labels of a categorical cross entropy.
    y_pred: A vector containing the predicted labels to the same categorical cross entropy.
    f_true: A vector containing the true labels of a binary crossentropy classification.
    f_pred: A vector containing the predicted labels of the binary crossentropy.

    Returns:
    A scalar representing the combined loss.
    """

    lambda_const = 0.5

    categorical_loss = np.mean((1 - f_true)*(1 - lambda_const) * (-np.sum(y_true * np.log(y_pred), axis=1)))

    binary_loss = lambda_const * np.mean(-f_true * np.log(f_pred) - (1 - f_true) * np.log(1 - f_pred))

    loss = categorical_loss + binary_loss


    return loss

images, y_true_1, y_true_2 = load_images()

model = create_cnn(images[0].shape, NUM_INPUT_IMAGES, 2)

opt = tf.keras.optimizers.SGD(
    learning_rate=1e-3,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4
)

model.compile(optimizer=opt,
              metrics={'output1': ['accuracy'], 'output2': ['accuracy']})

evaluation_metrics = model.fit([images, y_true_1, y_true_2], [y_true_1, y_true_2], batch_size=32)

for i, metric_name in enumerate(model.metrics_names):
    print(f"{metric_name}: {evaluation_metrics[i]}")

now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")

model.save(RESULT_FOLDER+'evaluated_model_' + dt_string + '.h5')
