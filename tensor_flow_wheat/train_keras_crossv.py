import math, requests
import os
from pathlib import Path
import tensorflow as tf

#from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model, utils
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import numpy as np
from tensorflow.keras.models import Sequential,load_model

#from keras import layers, metrics
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Flatten, Reshape,RandomFlip,RandomRotation,MaxPooling2D,Resizing,Rescaling,Conv2D)
#from keras.optimizers.legacy import adadelta
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
# additional callbacks and tflite functions
from sklearn.model_selection import KFold,StratifiedGroupKFold,StratifiedKFold
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
       
    # epochs and learning rate
    EPOCHS =  args.epochs
    LEARNING_RATE =  args.learning_rate
    #lr_list = [float(x) for x in args.lr_list.split(',')]
    lr_list=mf.generate_floats_around(LEARNING_RATE,4,0.0002)
    logging.info(f"lr_list {lr_list}")
    
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
    
    base_model_name=args.base_model
    include_top= args.include_top
    dropout = 0.2
    alpha = 1.3
    logging.info(f"args.include_top {args.include_top}")
    logging.info(f"include top layers ? --> {include_top}")
    if weights_path is None and include_top==True :
        logging.info("Not using pretrained weights. Training from scratch.")
        logging.info("Including top layers of base model")
        num_classes = args.num_classes
        logging.info(f"num_classes: {num_classes}")
        train_base_model = True
           
    else:
        num_classes = 1000
        train_base_model = False
        
    logging.info(f"base_model_name: {base_model_name}")
    
    base_model = mf.get_base_model(base_model_name,INPUT_SHAPE,
                                weights_path,include_top,num_classes,dropout,alpha,None)
        
        
    #logging.info(f"base_model , {base_model}")
    logging.info(f"Training base model ? --> {train_base_model}")
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
        monitor='val_sparse_categorical_accuracy',    # Monitor validation accuracy
        min_delta=min_d ,           # Minimum change to qualify as an improvement
        patience=patience,
        start_from_epoch=20,# Stop after 5 epochs without improvement
        verbose=1,                 # Print messages
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    ))

    
    # Define the K-fold Cross Validator
    num_folds = args.num_folds
    logging.info(f"Number of folds:{num_folds}")
    #kfold = KFold(n_splits=num_folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=num_folds,shuffle=True)

    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y")  
    print("Formatted date:", formatted_date)
    
    # save path is parent directory of both training checkpoints and keras models
    save_path =args.save_path
    checkpoint_path = os.path.join(save_path,"training_checkpoints",formatted_date,model_name)
    logging.info(f"checkpoint_path : {checkpoint_path}")
    
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #print("checkpoint_dir",checkpoint_dir)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    #model_callbacks.append(cp_callback)
    fold_no = 0
    results = {}
    for train, val in kfold.split(inputs, targets):
        results[fold_no] = {}
        logging.info(f"len(val) {len(val)}")
        logging.info(f"Training fold {fold_no} .....")
        # store checkpoint with fold number and epoch info
        #checkpoint=os.path.join(checkpoint_path,f"fold_{fold_no}"+"cp-{epoch:04d}.weights.h5")
        for lr in lr_list:
            base_model.trainable = False
            if lr <= 0:
                logging.error(f"Learning rate is too small ! : {lr} \n Continue ...")
                continue
            logging.info(f"Training with Learning rate {lr} ... \n")
            checkpoint=os.path.join(checkpoint_path,f"fold_{fold_no}_lr{lr}_"+"cp-best.weights.h5")
            print("checkpoint:",checkpoint)
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                        save_weights_only=True,
                                                        monitor="val_loss",
                                                        verbose=1,
                                                        save_freq="epoch",
                                                        save_best_only=True)
            # create model with prev. defined function
            model=mf.create_model(INPUT_SHAPE,base_model,num_classes,model_name)
            # compile
            model.compile(optimizer=Adam(learning_rate=lr),
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                        weighted_metrics=[SparseCategoricalAccuracy()])
            # fit model
            history = model.fit(
            inputs[train],
            targets[train],
            validation_data=(inputs[val],targets[val]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,    
            class_weight=weights,
            callbacks=[model_callbacks]
            )
            # fine tune 
            base_model.trainable = True
            base_model_layer_count = len(base_model.layers)
            # Let's take a look to see how many layers are in the base model
            print("Number of layers in the base model: ", len(base_model.layers))

            # Fine-tune from this layer onwards
            fine_tune_at = base_model_layer_count//2
            #Freeze all the layers before the `fine_tune_at` layer
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
                
            model.compile(optimizer=Adam(learning_rate=lr/10),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            weighted_metrics=[SparseCategoricalAccuracy()])
            
            trainable_var=len(model.trainable_variables)
            logging.info(f"{trainable_var} trainable variables for fine tuning")
            
            fine_tune_epochs = 15
            initial_epoch = len(history.epoch)
            total_epochs =  initial_epoch + fine_tune_epochs
            logging.info(f"Initial epoch for fine tuning : {initial_epoch}")
            history_fine = model.fit(train_ds,
                                    epochs=total_epochs,
                                    initial_epoch=initial_epoch,
                                    class_weight=weights,
                                    validation_data=(inputs[val],targets[val]),
                                    callbacks=[model_callbacks,cp_callback])
            
            # Generate generalization metrics
            scores = model.evaluate(inputs_test, targets_test, verbose=0)
            logging.info(f'Score for fold {fold_no} learning rate {lr}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
            accuracy = scores[1] * 100
            loss = scores[0]
            results[fold_no][lr] = {'accuracy': accuracy, 'loss': loss}
            
        # Increase fold number
        fold_no +=1
    logging.info(results)
    # Search for the best model
    best_fold = None
    best_lr = None
    best_accuracy = float('-inf')
    for fold, lr_dict in results.items():
        for lr, metrics in lr_dict.items():
            logging.info(f"LR {lr} : {metrics} \n")
            if metrics['accuracy'] > best_accuracy:  # Use accuracy or loss as criteria
                best_accuracy = metrics['accuracy']
                best_fold = fold
                best_lr = lr
    logging.info(f"fold {best_fold} has best accuracy with {results[best_fold][best_lr]}")
    
    # load best weights 
    best_weights_path = os.path.join(checkpoint_path,f"fold_{best_fold}_lr{best_lr}_"+"cp-best.weights.h5")
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