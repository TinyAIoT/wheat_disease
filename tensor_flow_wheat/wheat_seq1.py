import math, requests
from pathlib import Path
import tensorflow as tf
from tensorflow import Tensor
from keras.callbacks import EarlyStopping
from keras import Model, utils , callbacks
import random
import os
import numpy as np
import time
from keras.models import Sequential
from keras.applications import MobileNetV2
from keras import layers
from keras.layers import (
    Dense, InputLayer, Dropout, Conv1D, Flatten, Reshape, MaxPooling1D, BatchNormalization,
    Conv2D, GlobalMaxPooling2D, Lambda, GlobalAveragePooling2D)
from keras.optimizers.legacy import adam,adadelta
from keras.losses import categorical_crossentropy
import sys
from sklearn.utils import class_weight

class BatchLoggerCallback(tf.keras.callbacks.Callback):
    def __init__(self, batch_size, train_sample_count, epochs, interval_s = 10, ensure_determinism=False):
        # train_sample_count could be smaller than the batch size, so make sure total_batches is atleast
        # 1 to avoid a 'divide by zero' exception in the 'on_train_batch_end' callback.
        self.total_batches = max(1, int(train_sample_count / batch_size))
        self.last_log_time = time.time()
        self.epochs = epochs
        self.interval_s = interval_s
        print(f'Using batch size: {batch_size}', flush=True)

    # Within each epoch, print the time every 10 seconds
    def on_train_batch_end(self, batch, logs=None):
        current_time = time.time()
        if self.last_log_time + self.interval_s < current_time:
            print('Epoch {0}% done'.format(int(100 / self.total_batches * batch)), flush=True)
            self.last_log_time = current_time

    # Reset the time the start of every epoch
    def on_epoch_end(self, epoch, logs=None):
        self.last_log_time = time.time()

# Image dimensions
IMAGE_HEIGHT= 160
IMAGE_WIDTH= 160
# Input shape
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

#WEIGHTS_PATH = r"D:\TinyAIoT\Wheat_Disease\mobile_netv2_weights\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5"
WEIGHTS_PATH = './transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5'

# Download the model weights
root_url = 'https://cdn.edgeimpulse.com/'
p = Path(WEIGHTS_PATH)
if not p.exists():
    print(f"Pretrained weights {WEIGHTS_PATH} unavailable; downloading...")
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    weights_data = requests.get(root_url + WEIGHTS_PATH[2:]).content
    with open(WEIGHTS_PATH, 'wb') as f:
        f.write(weights_data)
    print(f"Pretrained weights {WEIGHTS_PATH} unavailable; downloading OK")
    print("")

# MobileNetV2
base_model = MobileNetV2(
    input_shape = INPUT_SHAPE, alpha=1,
    weights = WEIGHTS_PATH
)

base_model.trainable = False

# Implements the data augmentation policy
def augment_image(image, label):
    # Flips the image randomly
    image = tf.image.random_flip_left_right(image)

    # Vary the brightness of the image
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image, label


# Image size and batch size
IMAGE_SIZE = (160, 160)
BATCH_SIZE =  300

# get dataset from directory
set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"]
work_folder = r"D:\TinyAIoT\Wheat_Disease\dataset4\Long 2023 Plant Path 999 photos"
train_ds, val_ds = utils.image_dataset_from_directory(
    work_folder,
    "inferred",
    "int",
    set_class_names ,
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,  
)

# labels for getting class_names
y_train =[]
for images, labels  in train_ds.unbatch():
    y_train.append(labels.numpy())

# epochs and learning rate
EPOCHS =  7
LEARNING_RATE =  0.002
# unbatch first , maybe remove if not needed
train_ds = train_ds.unbatch()
# print images shape after unbacthing
print("\n Image shapes \n")
for images, labels  in train_ds:
    
    print(images.get_shape())
# Autotune
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#print("\n Image shapes after prefetch \n")
#for images, labels  in train_ds:
    
#    print(images.get_shape())

#class_names = train_ds.class_names
num_classes = len(set_class_names)
model = Sequential([
    #error  reshaping:
    #total size of new array must be unchanged, input_shape = [160, 160, 3], output_shape = [300, 160, 160, 3]
#layers.Reshape((BATCH_SIZE,160,160,3), input_shape=(160,160,3)),
layers.Rescaling(1./255, input_shape=INPUT_SHAPE),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
#model.add(InputLayer(input_shape=INPUT_SHAPE, name='x_input'))
# Don't include the base model's top layers
#last_layer_index = -3
#model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output,name="model1"))

for layer in model.layers:
    print(layer.name ,layer.input_shape, "---->", layer.output_shape)
# train_sample_count ?
train_sample_count = 800
# callbacks
model_callbacks =[]
# BatchLoggerCallback
model_callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


# apply early stopping
model_callbacks.append(EarlyStopping(
    monitor='val_accuracy',    # Monitor validation accuracy
    min_delta=0.01,           # Minimum change to qualify as an improvement
    patience=2,               # Stop after 2 epochs without improvement
    verbose=1,                 # Print messages
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
))

# compile
model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
#model.summary()
# class weights
print("np.unique(y_train):",np.unique(y_train))
sc_class_weights = class_weight.compute_class_weight('balanced',
                                                 classes= np.unique(y_train),
                                                y= y_train)
print("classweights",sc_class_weights)
#sc_class_weights = dict(zip(np.unique(y_train), sc_class_weights))
# fit

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS
)

print('')
print('Initial training done.', flush=True)

