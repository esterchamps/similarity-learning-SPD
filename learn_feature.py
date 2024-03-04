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

main_folder = 'preprocessed-images\\'
GENUINE_SIGNATURES_PREFIX = 'c-'
FORGED_SIGNATURES_PREFIX = 'cf-'
NUM_INPUT_IMAGES = 10
RESULT_FOLDER = 'training_results_2\\'

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

    label_layer_1 = tf.keras.layers.Input((num_classes1,))
    label_layer_2 = tf.keras.layers.Input((1,))

    model = tf.keras.Model(inputs=[input_layer, label_layer_1, label_layer_2], outputs=[output1, output2])

    lambda_const = tf.constant(0.5)

    categorical_loss = tf.reduce_mean(tf.math.negative(tf.math.multiply(tf.math.subtract(tf.constant(1.0), lambda_const), (tf.reduce_sum(tf.math.multiply(label_layer_1, tf.math.log(output1)), axis=1)))))

    binary_loss = tf.math.multiply(lambda_const, tf.reduce_mean(tf.math.subtract(tf.math.negative(tf.math.multiply(label_layer_2, tf.math.log(output2))), tf.math.multiply(tf.math.subtract(tf.constant(1.0), label_layer_2), tf.math.log(tf.math.subtract(tf.constant(1.0), output2))))))

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

            images.append(np.expand_dims((binary_image^1).astype(np.uint8), axis=-1))
            y_true_1.append(to_categorical(i-1, num_classes=NUM_INPUT_IMAGES))
            y_true_2.append(1 if img_file.startswith(FORGED_SIGNATURES_PREFIX) else 0)

            #data_structure.append({
            #    'image': otsu_image,
            #    'is_genuine': y_true_2[-1],
            #    'folder_number': y_true_1[-1]
            #})

    print('loaded images: '+str(len(images)))

    return np.array(images), np.array(y_true_1), np.array(y_true_2)#, data_structure

images, y_true_1, y_true_2 = load_images()

X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(images, y_true_1, y_true_2, test_size=0.1)

model = create_cnn(images[0].shape, NUM_INPUT_IMAGES, 2)

class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        if step.item() % 17920 == 0:
            self.initial_learning_rate /= 10
            return self.initial_learning_rate
        else:
            return self.initial_learning_rate


opt = tf.keras.optimizers.SGD(
    learning_rate=1e-3,
    momentum=0.9,
    nesterov=True
)

model.compile(optimizer=opt,
              metrics={'output1': ['accuracy'], 'output2': ['accuracy']})

evaluation_metrics = model.fit([X_train, y1_train, y2_train], [y1_train, y2_train], batch_size=32, epochs=60, validation_split=0.15, verbose=2)

test_results = model.evaluate([X_test, y1_train, y2_train],  [y1_test, y2_test], verbose=2)

print(str(model.metrics_names))
print('test results: '+str(test_results))

now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")

model.save(RESULT_FOLDER+'evaluated_model_' + dt_string + '.keras')
