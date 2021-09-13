# labeled image dataset with (x1, y1) (x2, y2) locations of unique template within image
# Convolutional Nerual Network that learns to output the position of the template within videoframe
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from PIL import Image, ImageDraw
from matplotlib import cm, pyplot as plt

# Load the Images and Labels
filenames = tf.data.Dataset.list_files("combined/*.jpg")
raw_labels = pd.read_csv('bounding_boxes.csv')[['0', '1', '2', '3']]

# raw_labels = raw_labels.div(10)

input_x = 295
input_y = 166

testname = tf.data.Dataset.list_files("*.jpg")

test_image = tf.io.decode_jpeg(tf.io.read_file(list(testname)[0]))
test_image = tf.image.resize(test_image, [input_x, input_y])

# Convert filenames to images, and resize each image so my laptop doesn't crash
unlabeled_images = filenames.map(lambda x: tf.io.decode_jpeg(tf.io.read_file(x)))
resized_images = unlabeled_images.map(lambda x: tf.image.resize(x, [input_x, input_y]))

# Split into training and test data
images_train = tf.stack(list(resized_images))
labels_train = tf.stack(raw_labels)

images_test = tf.stack(list(test_image))
images_test = tf.expand_dims(images_test, axis=0)
# labels_test = tf.stack(raw_labels[999])

# define a convolutional model
model = keras.models.Sequential()

# Input shape       (image.width x image.height, 3 color channels)
# Hidden Layers:    3 convolutional + pooling layers
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_x, input_y, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

# Conv to Dense:    Flatten & Dense layer
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))

# Output:           Dense (None, 4) layer as 4 corners of bounding box 
model.add(keras.layers.Dense(4))

# Compile the model
model.compile(  optimizer='adam',
                loss='mean_squared_error',
                metrics=['accuracy'])

# Get a list of all layers and layer_names
model.summary()

# Train the model
history = model.fit(images_train, 
                    labels_train,
                    epochs=5,
                    workers=1)

# Test the model
prediction = model.predict(images_test)

# # Draw the predictions onto the test image
# test_img = Image.open(str(list(testname)[0].numpy()).split('\\\\')[1][:-1])
# # test_img = Image.fromarray(np.uint8(list(unlabeled_images)[0])*255)
# draw_img = ImageDraw.Draw(test_img)
# draw_img.rectangle([(prediction[0][0], prediction[0][1]), (prediction[0][2], prediction[0][3])], width=10)
# test_img.save('test_results/test_result.jpg')

# Draw the output of an intermediate CNN layer when used to make a prediction on a test image
def visualize_CNN_layer(layer_name):
    # Get the layer argument from the trained model
    layer_output = model.get_layer(layer_name).output

    # Build and test an intermediate model
    middle_cnn = tf.keras.models.Model(inputs=model.input, outputs=layer_output)
    middle_prediction = middle_cnn.predict(images_test) # test_image

    frame_count = 64

    img_index = 0

    print(np.shape(middle_prediction))

    for img_index in range(frame_count):
            plt.imsave("temp/" + layer_name + "-" + str(img_index) + ".png", middle_prediction[0, :, :, img_index], format='png')

visualize_CNN_layer('conv2d_1')
visualize_CNN_layer('max_pooling2d_1')
visualize_CNN_layer('conv2d_2')

