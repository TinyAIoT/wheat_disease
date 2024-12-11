import tempfile
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow_model_optimization.python.core.keras.compat import keras

# %load_ext tensorboard

import tensorflow_datasets as tfds


pv_test, pv_train = tfds.load(name="plant_village",  split=["train[:20%]", "train[20%:100%]"], shuffle_files=True, batch_size=-1)
# tfds.as_numpy return a generator that yields NumPy array records out of a tf.data.Dataset
pv_test = tfds.as_numpy(pv_test)
pv_train = tfds.as_numpy(pv_train)

test_images, test_labels = pv_test["image"], pv_test["label"]
train_images, train_labels = pv_train["image"], pv_train["label"] # seperate the x and y

# plt.hist(pv_test['label'], bins=np.arange(38))  # arguments are passed to np.histogram


# Normalize the input image so that each pixel value is between 0 and 1.
# train_images = train_images / 255.0
# test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(256, 256, 3)),
  keras.layers.Reshape(target_shape=(256, 256, 3)),
  keras.layers.Rescaling(scale=1./255),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(38)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
  train_images,
  train_labels,
  epochs=4,
  validation_split=0.2,
)

_, baseline_model_accuracy = model.evaluate(
    test_images, test_labels, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)