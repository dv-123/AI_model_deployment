# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PLRtrlH0VTxhpO6SnoEoMmeqTtcBReaR
"""

#! pip install mnist

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mnist

train_images_1 = mnist.train_images()
train_labels_1 = mnist.train_labels()

test_images_1 = mnist.test_images()
test_labels_1 = mnist.test_labels()

train_images = np.expand_dims(train_images_1, axis = -1)
test_images = np.expand_dims(test_images_1, axis = -1)

train_labels = tf.keras.utils.to_categorical(train_labels_1)
test_labels = tf.keras.utils.to_categorical(test_labels_1)

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

plt.imshow(train_images_1[1000])
plt.show()
print(train_labels[1000])
print(train_labels_1[1000])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu', kernel_initializer='random_normal'))
classifier.add(Conv2D(32, (3, 3), activation = 'relu', kernel_initializer='random_normal'))
classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'SAME'))
classifier.add(Dropout(0.4))

classifier.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='random_normal'))
classifier.add(Conv2D(64, (3, 3), activation = 'relu', kernel_initializer='random_normal'))
classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'SAME'))
classifier.add(Dropout(0.4))

classifier.add(Flatten())

classifier.add(Dense(units = 128, activation = 'relu', kernel_initializer='random_normal'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 10, activation = 'softmax', kernel_initializer='random_normal'))

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001
)

classifier.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

classifier.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint('MNIST_digit.h5', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau()]

classifier.fit(train_images, train_labels, batch_size=64, epochs=100, verbose=1, validation_data = (test_images, test_labels), callbacks = callbacks)

MNIST_model = tf.keras.models.load_model('MNIST_digit.h5')

img = train_images[1010]
image = np.expand_dims(img, axis = 0)

prediction = MNIST_model.predict_classes(image)

prediction[0]

# saving the model
# serialize model to JSON
model_json = MNIST_model.to_json()
with open("model_json.json", "w") as json_file:
  json_file.write(model_json)
# serialize weights ro HDF5
MNIST_model.save_weights("model_weights.h5")
