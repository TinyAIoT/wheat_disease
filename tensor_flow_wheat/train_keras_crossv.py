import math, requests
import os
from pathlib import Path
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import Model, utils
from keras.metrics import SparseCategoricalAccuracy,Recall,Precision
import numpy as np
from keras.models import Sequential,load_model
from keras.applications import MobileNetV2
#from keras import layers, metrics
from keras.layers import (
    Dense, InputLayer, Dropout, Flatten, Reshape,RandomFlip,RandomRotation,Resizing,Rescaling,Conv2D)
#from keras.optimizers.legacy import adadelta
from keras.optimizers import Adam
from sklearn.utils import class_weight
# additional callbacks and tflite functions
from callbacks import BatchLoggerCallback
from sklearn.model_selection import KFold
import argparse
from datetime import datetime

# create model, compile
# change model_name if necessary

def create_model(input_shape,base_model,num_classes, plot_model:bool,model_name,image_height,image_width):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape, name='x_input'))
    # rescaling, rotation and flip
    model.add(RandomRotation(0.2))
    model.add(RandomFlip("horizontal_and_vertical"))
    model.add(Rescaling(1./127.5,offset=-1))
    # Don't include the base model's top layers
    last_layer_index = -3
    model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output))
    # reshape
    model.add(Reshape((-1, model.layers[-1].output.shape[3])))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # compile
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                weighted_metrics=[SparseCategoricalAccuracy()])
    return model

# Implements the data augmentation policy
def augment_image(image, label):
    
    # Vary the brightness of the image
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image, label


if __name__ == "__main__":
    # arguments for model training
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_folder', type=str, required=True, help='path to folder with validation and training data')
    parser.add_argument('--save_path', type=str, required=True, help='saving path for keras models and checkpoints')
    parser.add_argument('--pt_weights', type=str, required=False, help='pretrained weights for base model. If not available, will download weights')
    parser.add_argument('--model_name', type=str, required=True, help='name of model. Also defines saving path')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, required=True, help='learning rate')
    parser.add_argument('--patience', type=int, required=False, help='patience for early stopping')
    parser.add_argument('--min_delta', type=float, required=False, help='mind_delta for early stopping')
    parser.add_argument('--num_folds', type=int, required=False, help='number of folds for k-fold cross validation')
    parser.add_argument('--image_dim', type=int, required=False, help='image dimension x. if set input shape will be (x,x,3)')


    # parse arguments
    args = parser.parse_args()
        
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        print("GPU-Name:", gpu.name, "  Type:", gpu.device_type)

    # Image dimensions
    # path_name
    model_name =args.model_name
    print(f"Creating model {model_name} ... \n")
    data_folder = args.data_folder
    print(f"Using data from {data_folder} \n")
    print("----------")
    IMAGE_HEIGHT= args.image_dim or 480   
    IMAGE_WIDTH= args.image_dim or 480
    # Input shape
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    # mobile_net weights
    WEIGHTS_PATH = args.pt_weights or './transfer-learning-weights/keras/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160.h5'

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

    # batch size
    BATCH_SIZE =  args.batch_size
    # get dataset from directory
    set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"] 
    #"int": means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
    train_ds, test_ds = utils.image_dataset_from_directory(
        data_folder,
        "inferred", #labels are generated from the directory structure
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
        
    # class weights
    print("np.unique(y_train):",np.unique(y_train))
    weights = class_weight.compute_class_weight('balanced',
                                                    classes= np.unique(y_train),
                                                    y= y_train)
    weights = {i : weights[i] for i in range(5)}
    print("classweights",weights ,"\n")
    print("----------")
    
    # epochs and learning rate
    EPOCHS =  args.epochs
    LEARNING_RATE =  args.learning_rate
    print(f"Using learning-rate: {LEARNING_RATE} ")

    # Autotune
    AUTOTUNE = tf.data.AUTOTUNE
    # image augmentation
    train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    # prefetch
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    
    targets = np.concatenate([y for x, y in train_ds], axis=0)
    inputs = np.concatenate([x for x, y in train_ds], axis=0)
    
    #class_names = train_ds.class_names
    num_classes = len(set_class_names)
    print("num_classes",num_classes)
     
    # callbacks
    model_callbacks =[]
    # apply early stopping
    # min_delta parsed
    min_d = args.min_delta or 0.001
    print(f"min_delta for early stopping: {min_d} \n")
    # patience parsed
    patience=args.patience or 10
    print(f"patience for early stopping: {patience} \n")
    # append early stopping
    model_callbacks.append(EarlyStopping(
        monitor='loss',    # Monitor validation accuracy
        min_delta=min_d ,           # Minimum change to qualify as an improvement
        patience=patience,               # Stop after 5 epochs without improvement
        verbose=1,                 # Print messages
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    ))
    
    # Define the K-fold Cross Validator
    num_folds = args.num_folds
    print(f"Number of folds:{num_folds}")
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold =[]
    loss_per_fold=[]
    # save path is parent directory of both training checkpoints and keras models
    save_path =args.save_path
    checkpoint_path = os.path.join(save_path,"training_checkpoints",model_name)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("checkpoint_path",checkpoint_path)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    
    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    X_test = np.concatenate([x for x, y in test_ds], axis=0)
    # results = keras_model.evaluate(X_test, y_test, return_dict=True)

    #model_callbacks.append(cp_callback)
    for train, val in kfold.split(inputs, targets):
        
        # store checkpoint with fold number and epoch info
        #checkpoint=os.path.join(checkpoint_path,f"fold_{fold_no}"+"cp-{epoch:04d}.weights.h5")
        checkpoint=os.path.join(checkpoint_dir,f"fold_{fold_no}_"+"cp-best.weights.h5")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                    save_weights_only=True,
                                                    monitor="loss",
                                                    verbose=1,
                                                    save_freq="epoch",
                                                    save_best_only=True)
        # create model with prev. defined function
        model=create_model(INPUT_SHAPE,base_model,num_classes,False,model_name,IMAGE_HEIGHT,IMAGE_WIDTH)

        # fit model
        history = model.fit(
        inputs[train],
        targets[train],
        validation_data=(inputs[val], targets[val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,    
        class_weight=weights,
        callbacks=[model_callbacks,cp_callback]
        )
        # Generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Increase fold number
        fold_no = fold_no + 1
    # get index of best accuracy , +1 since fold number starts with 1
    best_accuracy = np.argmax(acc_per_fold)
    print(f"fold {best_accuracy+1} has best accuracy with {acc_per_fold[best_accuracy]}")
    # get index of best loss
    best_loss = np.argmin(loss_per_fold)
    print(f"fold {best_loss+1} has best loss value with {loss_per_fold[best_loss]}")
    
    # load best weights 
    best_weights_path = os.path.join(checkpoint_path,f"fold_{best_accuracy+1}_"+"cp-best.weights.h5")
    print("best weights in:",best_weights_path)
    model.load_weights(best_weights_path)
    
    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y")  
    print("Formatted date:", formatted_date)
    
    # path for saving
    keras_save_path= os.path.join(save_path,"keras_models",formatted_date,model_name,"model.keras")
    keras_save_dir = os.path.dirname(keras_save_path)
    if not os.path.exists(keras_save_dir):
        os.makedirs(keras_save_dir)
    # save keras model
    model.save(keras_save_path)
    
    print('')
    print('Initial training done. \n', flush=True)
    print('Keras model saved to .', keras_save_path, flush=True)