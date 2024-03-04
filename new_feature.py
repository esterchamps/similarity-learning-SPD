import tensorflow as tf
from image_utils import load_grayscale_image
import numpy as np
from skimage import filters

model = tf.keras.models.load_model('training_results_2\\evaluated_model_23022024_095700.h5')

oi = model.input

ola = model.layers[-2].output

intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                          outputs=model.layers[-2].output)

some_image = load_grayscale_image('preprocessed-images\\002\\cf-002-02.jpg')

threshold_value = filters.threshold_otsu(some_image)
binary_image = some_image > threshold_value
input = np.expand_dims((binary_image^1).astype(np.uint8), axis=-1)

input1 = np.ones((531,))

input2 = np.ones((1,))

intermediate_output = intermediate_layer_model([tf.expand_dims(input, 0), tf.expand_dims(input1, 0), tf.expand_dims(input2, 0)], training=False)
#.predict(input, input1, input2)

numpis = intermediate_output.numpy()

print(numpis)