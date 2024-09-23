import math, requests
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from pathlib import Path
import tensorflow as tf
from tensorflow import Tensor
from keras.callbacks import EarlyStopping
from keras import Model, utils , callbacks
import numpy as np
from keras.models import Sequential
from keras.applications import MobileNetV2
#from keras import layers, metrics
from keras.layers import (
    Dense, InputLayer, Dropout, Flatten, Reshape)
#from keras.optimizers.legacy import adadelta
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from sklearn.utils import class_weight
# additional callbacks and tflite functions
from callbacks import BatchLoggerCallback
from tflite_functions import convert_tflite_model,save_tflite_model,test_tflite


#tf.logging.set_verbosity(tf.logging.ERROR)
tf.get_logger().setLevel('WARN')
# create model, compile
# change model_name if necessary
def create_model(input_shape,base_model,num_classes, plot_model:bool,model_name):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape, name='x_input'))
    # Don't include the base model's top layers
    last_layer_index = -3
    model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output))
    model.add(Reshape((-1, model.layers[-1].output.shape[3])))

    # neurons and activation
    model.add(Dense(18, activation='relu'))
    # dropout
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # compile
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
    # plot model to png
    if plot_model:
        path = os.path.join(os.path.dirname(__file__),'training\models',model_name)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
            
        tf.keras.utils.plot_model(
                model,
                to_file=path+'/model.png',
                show_shapes=True,
            )
    # print layer shapes
    #for layer in model.layers:
    #    print(layer.name ,layer.input_shape, "---->", layer.output_shape)
    return model

# Image dimensions
# path_name
model_name = "mobile_net_v2"
data_folder = r"D:\TinyAIoT\Wheat_Disease\dataset4\Long 2023 Plant Path 999 photos"
IMAGE_HEIGHT= 160
IMAGE_WIDTH= 160
# Input shape
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
# mobile_net weights
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
     
    # Vary the brightness of the image
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image, label

# batch size
BATCH_SIZE =  128
# get dataset from directory
set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"]

#"inferred" (labels are generated from the directory structure)
#class_names 
#"int": means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
train_ds, val_ds = utils.image_dataset_from_directory(
    data_folder,
    "inferred",
    "int",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
    batch_size=BATCH_SIZE,  
)

# labels for getting class_names
y_train =[]
for images, labels  in train_ds.unbatch():
    y_train.append(labels.numpy())

# epochs and learning rate
EPOCHS =  1
LEARNING_RATE =  0.002
#print(train_ds)
# Autotune
AUTOTUNE = tf.data.AUTOTUNE
# image augmentation
#train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#class_names = train_ds.class_names
num_classes = len(set_class_names)
print("num_classes",num_classes)

# create model with prev. defined function
model=create_model(INPUT_SHAPE,base_model,num_classes,True,model_name)


# train_sample_count ?
print("card",train_ds.cardinality().numpy())

train_sample_count = train_ds.cardinality().numpy()
train_sample_count = math.ceil(train_sample_count) 
# callbacks
model_callbacks =[]
# BatchLoggerCallback
model_callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))
# apply early stopping
model_callbacks.append(EarlyStopping(
    monitor='val_accuracy',    # Monitor validation accuracy
    min_delta=0.01,           # Minimum change to qualify as an improvement
    patience=5,               # Stop after 5 epochs without improvement
    verbose=1,                 # Print messages
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
))
# Create a callback that saves the model's weights


checkpoint_path = os.path.join(os.path.dirname(__file__),"training_checkpoints_1\cp-{epoch:04d}.weights.h5")
checkpoint_dir = os.path.dirname(checkpoint_path)
print("checkpoint_path",checkpoint_path)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 monitor="val_loss",
                                                 verbose=1,
                                                 save_freq="epoch",
                                                 save_best_only=True)
model_callbacks.append(cp_callback)

model.summary()
# class weights
print("np.unique(y_train):",np.unique(y_train))
weights = class_weight.compute_class_weight('balanced',
                                                 classes= np.unique(y_train),
                                                y= y_train)
weights = {i : weights[i] for i in range(5)}
print("classweights",weights ,"\n")
print("----------")


# fit model
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=EPOCHS,    
  class_weight=weights,
  callbacks=model_callbacks
)
print(f"checkpoints in {checkpoint_dir} ",os.listdir(checkpoint_dir))

# path for saving
keras_save_path= os.path.join(os.path.dirname(__file__),"keras_"+model_name,"model.keras")
keras_save_dir = os.path.dirname(keras_save_path)

if not os.path.exists(keras_save_dir):
    os.makedirs(keras_save_dir)
# save keras model
model.save(keras_save_path)

# load model after saving
keras_model = tf.keras.models.load_model(keras_save_path)

# convert tflite
print("model to be converted:",keras_model)

tflite_model = convert_tflite_model(keras_model)

# save tflite
tflite_save_path = os.path.join(os.path.dirname(__file__),'training/models/'+model_name)
save_tflite_model(tflite_model, tflite_save_path, 'model.tflite')

print('')
print('Initial training done.', flush=True)

