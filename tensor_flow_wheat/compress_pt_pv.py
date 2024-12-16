import tempfile
import os

import tensorflow as tf
print(tf.__version__)
# import matplotlib.pyplot as plt
import numpy as np
import keras as ks
print(ks.__version__)
import argparse
import logging
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras
from tf_costum_utils import model_functions as mf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# hack to change model config from keras 2->3 compliant
from tensorflow.keras import layers

# %load_ext tensorboard

import tensorflow_datasets as tfds

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--model_name', type=str, required=True, help='name of model. Also defines saving path')
    parser.add_argument('--time', type=str, required=True, help='training start time')
    parser.add_argument('--save_path', type=str, required=True, help='name of output folder')
    parser.add_argument('--input_model', type=str, required=True, help='input_model_path')
    parser.add_argument('--training_data', type=str, required=True, help='path to training data')
    parser.add_argument('--test_ds_size', type=float, default=0.2, required=False, help='size of test set. If no value is given , the test set will hold 20 percent of the original data  ')
    parser.add_argument('--image_dim', type=int, required=False, help='image dimension x. if set input shape will be (x,x,3)')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--zen3', type=bool, required=False, help='Set True, if you are using zen3 for config')

    logging.basicConfig(level = logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s',datefmt='%H:%M:%S')

    # parse arguments
    args = parser.parse_args()
    zen3=args.zen3 or False
    if zen3:
       sess = mf.zen3_config()
    logging.info(f"Compressing model ... \n")
    print("----------")

    data_folder = args.training_data
    logging.info(f"Using data from {data_folder} \n")
    print("----------")
    test_ds_size=args.test_ds_size

    print(f"Loading model ... \n")
    print("----------")
    model = keras.models.load_model(args.input_model)

    print(f"Using data from {data_folder} \n")
    print("----------")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        "inferred", #labels are generated from the directory structure
        "categorical",
        validation_split=test_ds_size,
        subset='validation',
        seed=1337,
        image_size=(224,224),
        batch_size=args.batch_size
    )
    
  # Configure variables for Transfer learning
    image_size = 224
    target_size = (image_size, image_size)
    input_shape = (image_size, image_size, 3)
    grid_shape = (1, image_size, image_size, 3)


    AUTOTUNE = tf.data.AUTOTUNE
    # # image augmentation
    # prefetch
    train_ds = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

    targets = np.concatenate([y for x, y in normalized_ds], axis=0)
    inputs = np.concatenate([x for x, y in normalized_ds], axis=0)

    _, baseline_model_accuracy = model.evaluate(
       inputs, targets, verbose=0)


    print('Baseline test accuracy:', baseline_model_accuracy)

    _, keras_file = tempfile.mkstemp('.h5')

    
    keras.models.save_model(model, keras_file, include_optimizer=False)
    print('Saved baseline model to:', keras_file)
    model.summary()

    def get_gzipped_model_size(file):
      # Returns size of gzipped model, in bytes.
      import os
      import zipfile

      _, zipped_file = tempfile.mkstemp('.zip')
      with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

      return os.path.getsize(zipped_file)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_and_pruned_tflite_model = converter.convert()

    quantized_and_pruned_tflite_file = os.path.join(args.save_path,"pv_quant.tflite")

    with open(quantized_and_pruned_tflite_file, 'wb') as f:
      f.write(quantized_and_pruned_tflite_model)
  
    # _, quantized_and_pruned_tflite_file = tempfile.mkstemp('.tflite')

    # with open(quantized_and_pruned_tflite_file, 'wb') as f:
    #   f.write(quantized_and_pruned_tflite_model)

    print('Saved quantized and pruned TFLite model to:', quantized_and_pruned_tflite_file)

    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(keras_file)))
    print("Size of gzipped pruned and quantized TFlite model: %.2f bytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file)))
    print("Size of gzipped baseline Keras model: %.2f Mbytes" % (get_gzipped_model_size(keras_file) / 1000000))
    print("Size of gzipped pruned and quantized TFlite model: %.2f Mbytes" % (get_gzipped_model_size(quantized_and_pruned_tflite_file) / 1000000))


    def evaluate_model(interpreter):
      input_index = interpreter.get_input_details()[0]["index"]
      output_index = interpreter.get_output_details()[0]["index"]

      # Run predictions on ever y image in the "test" dataset.
      prediction_digits = []
      prediction_targets = []
      i = 0
      for i, test_image in enumerate(inputs):
        if i % 1000 == 0:
          print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        target = np.argmax(targets[i])
        prediction_digits.append(digit)
        prediction_targets.append(target)

      print('\n')
      # Compare prediction results with ground truth labels to calculate accuracy.
      prediction_digits = np.array(prediction_digits)
      prediction_targets = np.array(prediction_targets)
      accuracy = (prediction_digits == prediction_targets).mean()
      return accuracy

    interpreter = tf.lite.Interpreter(model_content=quantized_and_pruned_tflite_model)
    interpreter.allocate_tensors()

    test_accuracy = evaluate_model(interpreter)

    print('Pruned and quantized TFLite test_accuracy:', test_accuracy)
    print('Baseline test accuracy:', baseline_model_accuracy)