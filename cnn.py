import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import matplotlib.pyplot as plt
import paths, utils
import PIL

model = models.Sequential()

trainimgs = []
trainlabels = np.array([])

testimgs = []
tests = paths.test_paths
for i in tests:
  img = load_img("source/angler/2016-03-21-other-Angler-EK-landing-page-after-localtasteblog.com.exe_IMGP.png")
  img_array = utils.arrayops(img)

  testimgs.append(img_array)
testlabels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

angler = paths.angler_paths
for i in angler:
  path = "source/angler/" + i
  img = load_img(path)
  img_array = utils.arrayops(img)

  trainimgs.append(img_array)
  np.append(trainlabels, np.array([0, 1]))

benign = paths.benign_paths
for i in benign:
  path = "source/benign/" + i
  img = load_img(path)
  img_array = utils.arrayops(img)

  trainimgs.append(img_array)
  np.append(trainlabels, np.array([1, 0]))

# Defining CNN model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adding dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# history = model.fit(trainimgs, trainlabels, epochs=10, 
#                     validation_data=(testimgs, testlabels))
utils.model()
