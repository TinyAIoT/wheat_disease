# import tempfile
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
import argparse

# %load_ext tensorboard

import tensorflow_datasets as tfds

if __name__ == "__main__":
    # arguments for model training
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', type=str, required=True, help='name of model. Also defines saving path')
    parser.add_argument('--time', type=str, required=True, help='training start time')
    parser.add_argument('--out', type=str, required=True, help='name of output folder')
    parser.add_argument('--base_model', type=str, required=True, help='input_model_path')
    
    # parse arguments
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Shrinking model {model_name} ... \n")
    print("----------")


_URL = 'https://uni-muenster.sciebo.de/s/Elqpf7UFF8NDQed/download'
path_to_zip = tf.keras.utils.get_file('plants.zip', origin=_URL, extract=True, cache_dir='/scratch/tmp/b_kari02/data/')
PATH = os.path.join(os.path.dirname(path_to_zip), 'plants')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE)

class_names = train_dataset.class_names
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


model = tf.keras.models.load_model(args.base_model)
# _, baseline_model_accuracy = model.evaluate(
#     test_images, test_labels, verbose=0)

# print('Baseline test accuracy:', baseline_model_accuracy)
loss, accuracy, recall, precision = model.evaluate(test_dataset)
print('Test loss :', loss)
print('Test recall :', recall)
print('Test precision :', precision)
print('Test accuracy :', accuracy)


keras_file = args.out + 'keras_file.h5'
keras.models.save_model(model, keras_file, include_optimizer=False)
print('Saved baseline model to:', keras_file)

model.summary()


import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 32
epochs = 2
validation_split = 0.1 # 10% of training set will be used for validation set.

end_step_count = tf.data.experimental.cardinality(train_dataset) * epochs

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
          initial_sparsity=0.0,   # Start from no sparsity
          final_sparsity=0.5,     # Aim to reach 50% sparsity
          begin_step=0,           # Start pruning immediately
          end_step=end_step_count,           # End pruning after roughly 4 epochs
          frequency=100           # Update sparsity every 100 steps
  )
}

# Helper function uses `prune_low_magnitude` to make only the 
# Dense layers train with pruning.
def apply_pruning_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Conv2D):
    print("Found a Conv2D Layer I can prune!")
    return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
  print("Not a Conv2D layer")
  return layer



data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
])
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid')

# Creating a new model with pruned layers, respecting the functional API
inputs = tf.keras.Input(shape=(160, 160, 3))
x = inputs
output_layer_map = {}

for layer in model.layers:
    print(layer.name)
    print(layer)
    if 'efficientnetv2-s' in layer.name:   # Processing within EfficientNet
        # if layer.name not in output_layer_map:
            # Apply pruning if it’s a convolutional layer
            # pruned_layer = apply_pruning_to_dense(layer)
            # Connect it appropriately to previous layers
      if isinstance(layer, tf.keras.Model):
        for sublayer in layer.layers:
            # if layer.name == 'efficientnetv2-s':
            #     # For the first EfficientNet layer, the input is directly x
          pruned_layer = apply_pruning_to_dense(sublayer)
            # else:
                # For subsequent layers, the input is the output of some other layer(s)
          print(sublayer)
          inbound_nodes = sublayer.inbound_nodes[0].inbound_layers
          if isinstance(inbound_nodes, list):
              inputs = [output_layer_map[l.name] for l in inbound_nodes]
          else:
              inputs = output_layer_map[inbound_nodes.name]
          print(inputs)
          print(inbound_nodes)
          x = pruned_layer(inputs)
          output_layer_map[sublayer.name] = x
    else:
        pruned_layer = apply_pruning_to_dense(layer)
        x = pruned_layer(x)
        # Outside EfficientNet processing
        # ... similar handling for connecting layers correctly
        pass


# Build final model
final_pruned_model = tf.keras.Model(inputs, x)

# Compile, fit, etc., as before


# def apply_pruning_to_all_but_rescaling(layer):
#   if not isinstance(layer, keras.layers.Rescaling):
#     return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
#   return layer

# Use `keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
# model_for_pruning = keras.models.clone_model(
#     model,
#     clone_function=apply_pruning_to_dense)

# model_for_pruning = prune_low_magnitude(model, **pruning_params)
base_learning_rate = 0.0001

# `prune_low_magnitude` requires a recompile.
final_pruned_model.compile(tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
                        tf.keras.metrics.Recall(),
                        tf.keras.metrics.Precision()])


final_pruned_model.summary()

logdir = args.out

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

final_pruned_model.fit(train_dataset,
                  batch_size=batch_size, epochs=epochs, validation_data=validation_dataset,
                  callbacks=callbacks)

pruned_loss, pruned_accuracy, pruned_recall, pruned_precision = final_pruned_model.evaluate(test_dataset)
print('Test loss :', loss)
print('Test recall :', recall)
print('Test precision :', precision)
print('Test accuracy :', accuracy)


print('Baseline test accuracy:', accuracy)
print('Pruned test accuracy:', pruned_accuracy)

model_for_export = tfmot.sparsity.keras.strip_pruning(final_pruned_model)

pruned_keras_file = args.out + "pruned_keras_file.h5"
keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
print('Saved pruned Keras model to:', pruned_keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
pruned_tflite_model = converter.convert()

pruned_tflite_file = args.out+'.tflite'

with open(pruned_tflite_file, 'wb') as f:
  f.write(pruned_tflite_model)

print('Saved pruned TFLite model to:', pruned_tflite_file)

def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  zipped_file = args.out+'zipped_file.zip'
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
print("Size of gzipped baseline Keras model: %.2f Mbytes" % (get_gzipped_model_size(keras_file) / 1000000))
print("Size of gzipped pruned Keras model: %.2f Mbytes" % (get_gzipped_model_size(pruned_keras_file) / 1000000))
print("Size of gzipped pruned TFlite model: %.2f Mbytes" % (get_gzipped_model_size(pruned_tflite_file) / 1000000))

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_and_pruned_tflite_model = converter.convert()

quantized_and_pruned_tflite_file = args.out + 'quantized_and_pruned_tflite_file.tflite'

with open(quantized_and_pruned_tflite_file, 'wb') as f:
  f.write(quantized_and_pruned_tflite_model)

print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
print("Size of gzipped baseline Keras model: %.2f Mbytes" % (get_gzipped_model_size(keras_file) / 1000000))
print("Size of gzipped pruned and quantized TFlite model: %.2f Mbytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file) / 1000000))


# interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
# interpreter.allocate_tensors()

# # loss, test_accuracy, recall, precision = quantized_and_pruned_tflite_model.evaluate(test_dataset)

# quantize_model = tfmot.quantization.keras.quantize_model
# q_aware_model = quantize_model(model_for_export)

# q_aware_model.compile(tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
#               loss=keras.losses.BinaryCrossentropy(),
#               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy'),
#                         tf.keras.metrics.Recall(),
#                         tf.keras.metrics.Precision()])

# q_aware_model.summary()

# q_aware_model.fit(train_dataset, validation_data=validation_dataset, epochs=5)
# qloss, qtest_accuracy, qrecall, qprecision = q_aware_model.evaluate(test_dataset)

# print('Pruned and quantized TFLite test_accuracy:', qtest_accuracy)
# print('Pruned TF test accuracy:', pruned_accuracy)
# print('Pruned loss, recall, precision:', qloss, ' , ', qrecall, ' , ', qprecision)

# converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# tflite_qaware_model = converter.convert()

# tflite_qaware_model_file = args.out + 'tflite_qaware_model_file.tflite'

# with open(tflite_qaware_model_file, 'wb') as f:
#   f.write(tflite_qaware_model)

# print('Saved quantized and pruned TFLite model to:', tflite_qaware_model_file)

# print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(tflite_qaware_model_file)))
# print("Size of gzipped pruned and quantized TFlite model: %.2f Mbytes" % (get_gzipped_model_size(tflite_qaware_model_file) / 1000000))
