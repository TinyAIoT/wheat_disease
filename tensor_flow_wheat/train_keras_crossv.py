import math, requests
import os
from pathlib import Path
import tensorflow as tf

#from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model, utils
from tensorflow.keras.metrics import SparseCategoricalAccuracy,Recall,Precision
import numpy as np
from tensorflow.keras.models import Sequential,load_model

#from keras import layers, metrics
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Flatten, Reshape,RandomFlip,RandomRotation,MaxPooling2D,Resizing,Rescaling,Conv2D)
#from keras.optimizers.legacy import adadelta
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
# additional callbacks and tflite functions
from sklearn.model_selection import KFold,StratifiedGroupKFold
import argparse
from datetime import datetime
import logging
from tf_costum_utils.callbacks import (BatchLoggerCallback,
LearningRatePrinter,get_callback_list,get_lr_scheduler)
from tf_costum_utils import model_functions as mf


if __name__ == "__main__":
    # arguments for model training
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_folder', type=str, required=True, help='path to folder with validation and training data')
    parser.add_argument('--base_model', type=str, required=False, help='Set base model. e.g mobile_net_v2 or mobile_net_3')
    parser.add_argument('--include_top',default=False, type=bool, required=False, help='True if top layers of base model should be included')
    parser.add_argument('--save_path', type=str, required=True, help='saving path for keras models and checkpoints')
    parser.add_argument('--pt_weights',default=None, type=str, required=False, help='pretrained weights for base model. If not available, will download weights')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--model_name', type=str, required=True, help='name of model. Also defines saving path')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, required=True, help='learning rate')
    parser.add_argument('--patience', type=int, required=False, help='patience for early stopping')
    parser.add_argument('--min_delta', type=float, required=False, help='mind_delta for early stopping')
    parser.add_argument('--num_folds', type=int, required=False, help='number of folds for k-fold cross validation')
    parser.add_argument('--image_dim', type=int, required=False, help='image dimension x. if set input shape will be (x,x,3)')
    parser.add_argument('--zen3', type=bool, required=False, help='Set True, if you are using zen3 for config')
    parser.add_argument('--test_ds_size', type=float, default=0.2, required=False, help='size of test set. If no value is given , the test set will hold 20 percent of the original data  ')

    logging.basicConfig(level = logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s',datefmt='%H:%M:%S')

    # parse arguments
    args = parser.parse_args()
    zen3=args.zen3 or False
    if zen3:
       sess = mf.zen3_config()
        
    # parsed model name
    model_name =args.model_name
    logging.info(f"Creating model {model_name} ... \n")
    # data folder
    data_folder = args.data_folder
    logging.info(f"Using data from {data_folder} \n")
    print("----------")
    # Image Dimsension
    IMAGE_HEIGHT= args.image_dim or 224
    IMAGE_WIDTH= args.image_dim or 224
    # Input shape
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    logging.info(f"specified INPUT_SHAPE: {INPUT_SHAPE}")
    # mobile_net weights
    weights_path = args.pt_weights or None
    logging.info(f"WEIGHTS_PATH : {weights_path} \n",)
    
    #print("weights",model_weights)
   
    #class_names = train_ds.class_names
    # get dataset from directory
    #set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"] 
    num_classes = args.num_classes
    logging.info(f"num_classes: {num_classes}")
    
    base_model_name=args.base_model
    include_top= args.include_top
    dropout = 0.2
    alpha = 1.0
    logging.info(f"include top layers ? --> {include_top}")
    if weights_path is None:
        logging.info("Not using pretrained weights. Training from scratch.")
        train_base_model = True
           
    if include_top == True and weights_path=="imagenet":
        logging.info("Setting number of classes to 1000 because include_top = True and weights = 'imagenet' ")
        num_classes = 1000
        train_base_model = False
        
    logging.info(f"base_model_name: {base_model_name}")
    
    base_model = mf.get_base_model(base_model_name,INPUT_SHAPE,
                                weights_path,include_top,num_classes,dropout,alpha,None)
        
        
    #logging.info(f"base_model , {base_model}")
    base_model.trainable = train_base_model

    # batch size
    BATCH_SIZE =  args.batch_size
    test_ds_size=args.test_ds_size
    #"int": means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
    train_ds, test_ds = utils.image_dataset_from_directory(
        data_folder,
        "inferred", #labels are generated from the directory structure
        "int",
        validation_split=test_ds_size,
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
    logging.info(f"classweights: {weights} \n")
    
    # epochs and learning rate
    EPOCHS =  args.epochs
    LEARNING_RATE =  args.learning_rate
    logging.info(f"Using learning-rate: {LEARNING_RATE} ")

    # Autotune
    AUTOTUNE = tf.data.AUTOTUNE
    # image augmentation
    train_ds = train_ds.map(mf.augment_image, num_parallel_calls=AUTOTUNE)
    # prefetch
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    targets = np.concatenate([y for x, y in train_ds], axis=0)
    inputs = np.concatenate([x for x, y in train_ds], axis=0)
    
    targets_test = np.concatenate([y for x, y in test_ds], axis=0)
    inputs_test = np.concatenate([x for x, y in test_ds], axis=0)
    
    # callbacks
    model_callbacks =[]
    # apply early stopping
    # min_delta parsed
    min_d = args.min_delta or 0.001
    logging.info(f"min_delta for early stopping: {min_d} \n")
    # patience parsed
    patience=args.patience or 10
    logging.info(f"patience for early stopping: {patience} \n")
    # append early stopping
    model_callbacks.append(EarlyStopping(
        monitor='loss',    # Monitor validation accuracy
        min_delta=min_d ,           # Minimum change to qualify as an improvement
        patience=patience,               # Stop after 5 epochs without improvement
        verbose=1,                 # Print messages
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    ))
    model_callbacks.append(LearningRatePrinter())
    
    # Define the K-fold Cross Validator
    num_folds = args.num_folds
    logging.info(f"Number of folds:{num_folds}")
    #kfold = KFold(n_splits=num_folds, shuffle=True)
    kfold = StratifiedGroupKFold(n_splits=num_folds,shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold =[]
    loss_per_fold=[]
    # save path is parent directory of both training checkpoints and keras models
    
    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y")  
    print("Formatted date:", formatted_date)
    
    save_path =args.save_path
    checkpoint_path = os.path.join(save_path,"training_checkpoints",formatted_date,model_name)
    logging.info(f"checkpoint_path : {checkpoint_path}")
    
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #print("checkpoint_dir",checkpoint_dir)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    #model_callbacks.append(cp_callback)
    for train, val in kfold.split(inputs, targets):
        logging.info(f"Training fold {fold_no} .....")
        # store checkpoint with fold number and epoch info
        #checkpoint=os.path.join(checkpoint_path,f"fold_{fold_no}"+"cp-{epoch:04d}.weights.h5")
        checkpoint=os.path.join(checkpoint_path,f"fold_{fold_no}_"+"cp-best.weights.h5")
        print("checkpoint:",checkpoint)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                    save_weights_only=True,
                                                    monitor="loss",
                                                    verbose=1,
                                                    save_freq="epoch",
                                                    save_best_only=True)
        
        #tboard_path = os.path.join(save_path,f'logs/{fold_no}')
        #print(tboard_path)
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tboard_path, histogram_freq=1)
        # create model with prev. defined function
        lr_printer=LearningRatePrinter()
        model=mf.create_model(INPUT_SHAPE,base_model,LEARNING_RATE,num_classes,False,model_name)
        lr_scheduler = get_lr_scheduler()
        #callback_list=get_callback_list([model_callbacks,cp_callback,lr_scheduler,lr_printer],model)
        # fit model
        history = model.fit(
        inputs[train],
        targets[train],
        validation_data=(inputs[val],targets[val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,    
        class_weight=weights,
        callbacks=[model_callbacks,cp_callback,lr_scheduler,lr_printer]
        )
        
        # Generate generalization metrics
        scores = model.evaluate(inputs_test, targets_test, verbose=0)
        logging.info(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Increase fold number
        fold_no +=1
    # get index of best accuracy , +1 since fold number starts with 1
    best_accuracy = np.argmax(acc_per_fold)
    logging.info(f"fold {best_accuracy+1} has best accuracy with {acc_per_fold[best_accuracy]}")
    # get index of best loss
    best_loss = np.argmin(loss_per_fold)
    logging.info(f"fold {best_loss+1} has best loss value with {loss_per_fold[best_loss]}")
    
    # load best weights 
    best_weights_path = os.path.join(checkpoint_path,f"fold_{best_accuracy+1}_"+"cp-best.weights.h5")
    logging.info(f"best weights in: {best_weights_path}")
    model.load_weights(best_weights_path)
    
    
    
    # path for saving
    keras_save_path= os.path.join(save_path,"keras_models",formatted_date,model_name,"model.keras")
    keras_save_dir = os.path.dirname(keras_save_path)
    if not os.path.exists(keras_save_dir):
        os.makedirs(keras_save_dir)
    # save keras model
    model.save(keras_save_path)

    logging.info('Initial training done. \n')
    logging.info(f'Keras model saved to {keras_save_path}.')