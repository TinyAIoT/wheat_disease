import math, requests
import os
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV2
#from keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model, utils
from tensorflow.keras.metrics import SparseCategoricalAccuracy,Recall,Precision
import numpy as np
from tensorflow.keras.models import Sequential,load_model

#from keras import layers, metrics
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, GlobalAveragePooling2D, Flatten, Reshape,RandomFlip,RandomRotation,MaxPooling2D,Resizing,Rescaling,Conv2D)
#from keras.optimizers.legacy import adadelta
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
# additional callbacks and tflite functions
from callbacks import BatchLoggerCallback
from sklearn.model_selection import KFold,StratifiedGroupKFold,StratifiedKFold
import argparse
from datetime import datetime

# create model, compile
# change model_name if necessary

def select_basemodel(name):
    if name == "mobile_net_v2":
        print("mobile_net_v2 as base model ... \n")
        # MobileNetV2
        # If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000.
        return tf.keras.applications.MobileNetV2(
            input_shape = INPUT_SHAPE,
            # alpha=1,
            weights = 'imagenet',
            include_top = False,
            include_preprocessing=False,
            # classes=num_classes, # only to be specified if include_top is True, and if no weights argument is specified
            #pooling="avg" 
        )
        
    if name == "mobile_net_v3":
        print("mobile_net_v3 as base model ... \n")
        # MobileNetV3
        return tf.keras.applications.MobileNetV3Small(
            input_shape = INPUT_SHAPE,
            # alpha=1,
            # dropout_rate=0.2,
            weights = 'imagenet',
            include_top = False,
            include_preprocessing=False
            #classes=num_classes, # only to be specified if include_top is True, and if no weights argument is specified
        )
    
    else:
        # fallback to Mobilenetv3 for now
        return tf.keras.applications.MobileNetV3Small(
            input_shape = INPUT_SHAPE,
            # alpha=1,
            # dropout_rate=0.2,
            weights = 'imagenet',
            include_top = False,
            include_preprocessing=False
            #classes=num_classes, # only to be specified if include_top is True, and if no weights argument is specified
        )



if __name__ == "__main__":
    # arguments for model training
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_folder', type=str, required=True, help='path to folder with validation and training data')
    parser.add_argument('--base_model', type=str, required=False, help='Set base model. e.g mobile_net_v2 or mobile_net_3')
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


    # parse arguments
    args = parser.parse_args()
    zen3=args.zen3 or False
    if zen3:
        #config = tf.ConfigProto()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)
        sess = tf.compat.v1.Session(config=config)
        
    # parsed model name
    model_name =args.model_name
    print(f"Creating model {model_name} ... \n")
    # data folder
    data_folder = args.data_folder
    print(f"Using data from {data_folder} \n")
    print("----------")
    # Image Dimsension
    IMAGE_HEIGHT= args.image_dim or 224
    IMAGE_WIDTH= args.image_dim or 224
    # Input shape
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    # mobile_net weights
    # weights_path = args.pt_weights or None
    # print("WEIGHTS_PATH : \n",weights_path)
    # if weights_path == None:
    #     weights="imagenet"
    # else:
    #     weights=weights_path
   
    #class_names = train_ds.class_names
    # get dataset from directory
    #set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"] 
    num_classes = args.num_classes
    print("num_classes",num_classes)
    
    base_model_name=args.base_model
    print("base_model_name",base_model_name)
    base_model = select_basemodel(base_model_name)
    print("base_model",base_model)
    base_model.trainable = False

    # Get the current date and time
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y")  
    print("Formatted date:", formatted_date)
    
    save_path =args.save_path
    checkpoint_path = os.path.join(save_path,"training_checkpoints",formatted_date,model_name)
    print("checkpoint_path :",checkpoint_path)
    
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #print("checkpoint_dir",checkpoint_dir)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # batch size
    BATCH_SIZE =  args.batch_size
    
    #"int": means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
    train_ds, val_ds = utils.image_dataset_from_directory(
        data_folder,
        "inferred", #labels are generated from the directory structure
        "int",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
        batch_size=BATCH_SIZE,  
    )
    # # labels for getting class_names
    # y_train =[]
    # for images, labels  in train_ds.unbatch():
    #     y_train.append(labels.numpy())
        
    # class weights
    # print("np.unique(y_train):",np.unique(y_train))
    # weights = class_weight.compute_class_weight('balanced',
    #                                                 classes= np.unique(y_train),
    #                                                 y= y_train)
    # weights = {i : weights[i] for i in range(5)}
    # print("classweights",weights ,"\n")
    # print("----------")
    
    # epochs and learning rate
    EPOCHS =  args.epochs
    LEARNING_RATE =  args.learning_rate
    print(f"Using learning-rate: {LEARNING_RATE} ")

    # Autotune
    AUTOTUNE = tf.data.AUTOTUNE
    
    resize_and_rescale = tf.keras.Sequential([
        Resizing(IMAGE_HEIGHT, IMAGE_WIDTH),
        Rescaling(1./127.5,offset=-1)
    ])

    # Implements the data augmentation policy
    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(0.2),
    ])
        
    def prepare(ds, shuffle=False, augment=False):
    # Resize and rescale all datasets.
        ds = ds.map(lambda x, y: (resize_and_rescale(x), y), 
                    num_parallel_calls=AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(1000)

        # Batch all datasets.
        # ds = ds.batch(BATCH_SIZE)

        # Use data augmentation only on the training set.
        if augment:
            ds = ds.map(lambda x, y: (data_augmentation(x), y), 
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)

    train_ds = prepare(train_ds, shuffle=False, augment=True)
    val_ds = prepare(val_ds)
    # test_ds = prepare(test_ds)
    
    # image augmentation
    # train_ds = train_ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    # prefetch
    # train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    inputs = np.concatenate(list(train_ds.map(lambda x, y:x)))
    targets = np.concatenate(list(train_ds.map(lambda x, y:y)))

    val_inputs = np.concatenate(list(val_ds.map(lambda x, y:x)))
    val_targets = np.concatenate(list(val_ds.map(lambda x, y:y)))

    # inputs = np.concatenate((train_images, val_images), axis=0)
    # targets = np.concatenate((train_labels, val_labels), axis=0)

    # inputs = np.concatenate((train_images), axis=0)
    # targets = np.concatenate((train_labels), axis=0)
    
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
    
    # inputs = np.concatenate([x for x, y in train_ds], axis=0)
    # targets = np.concatenate([y for x, y in train_ds], axis=0)
    # targets_num = np.concatenate([y for x, y in train_ds], axis=0)
    print(targets)
    # print(targets_num)
    # inputs_test = np.concatenate([x for x, y in test_ds], axis=0)
    # targets_test = np.concatenate([y for x, y in test_ds], axis=0)
    
    # Define the K-fold Cross Validator
    num_folds = args.num_folds
    print(f"Number of folds:{num_folds}")
    # kfold = KFold(n_splits=num_folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=num_folds,shuffle=True)
    print(kfold.get_n_splits(inputs, targets))
    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold =[]
    loss_per_fold=[]
    
    plt.figure(figsize=(6*num_folds, 8))
    for train_index, test_index in kfold.split(inputs, targets):
        print(test_index)
        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        # store checkpoint with fold number and epoch info
        checkpoint=os.path.join(checkpoint_path,f"fold_{fold_no}_"+"cp-best.weights.h5")
        print("checkpoint:",checkpoint)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                    save_weights_only=True,
                                                    monitor="loss",
                                                    verbose=1,
                                                    save_freq="epoch",
                                                    save_best_only=True)
        # create model with prev. defined function
        model = Sequential()
        model.add(InputLayer(input_shape=INPUT_SHAPE, name='x_input'))
        # model.add(Rescaling(1./127.5,offset=-1))
        model.add(Model(inputs=base_model.inputs, outputs=base_model.output))
        model.add(GlobalAveragePooling2D())
        # model.add(Reshape((-1, model.layers[-1].output.shape[3])))
        model.add(Dropout(0.2))
        # model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax'))
        
        # compile
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(name='loss'),
                    weighted_metrics=[SparseCategoricalAccuracy(name='accuracy')])
        
        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')
        
        # fit model
        history = model.fit(
            X_train,
            y_train,
            # validation_data=(X_test,y_test),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,    
            # class_weight=weights,
            callbacks=[model_callbacks,cp_callback])
        
        acc = history.history['accuracy']
        # val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        # val_loss = history.history['val_loss']

        plt.subplot(2, num_folds, fold_no)
        plt.plot(acc, label='Training Accuracy')
        # plt.plot(val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, num_folds, fold_no+num_folds)
        plt.plot(loss, label='Training Loss')
        # plt.plot(val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        plt.ylim([0,100.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
                
        # Generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(scores)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Increase fold number
        fold_no = fold_no + 1
    
    plt.savefig(os.path.join(save_path,"loss_curve_fold"+str(num_folds)+formatted_date+model_name+".png"))        
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')
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
    
    
    
    # path for saving
    keras_save_path= os.path.join(save_path,"keras_models",formatted_date,model_name,"model.keras")
    keras_save_dir = os.path.dirname(keras_save_path)
    if not os.path.exists(keras_save_dir):
        os.makedirs(keras_save_dir)
    # save keras model
    model.save(keras_save_path)
    print('------------------------------------------------------------------------')
    print('Top Score Pre Tuning:')
    print(scores)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    print('------------------------------------------------------------------------')

    base_model.trainable = True
    
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    
    fine_tune_epochs = 10
    total_epochs =  EPOCHS + fine_tune_epochs

    num_folds = args.num_folds
    print(f"Number of folds:{num_folds}")
    # kfold = KFold(n_splits=num_folds, shuffle=True)
    kfold = StratifiedKFold(n_splits=num_folds,shuffle=True)
    print(kfold.get_n_splits(inputs, targets))
    # K-fold Cross Validation model evaluation
    fold_no = 1
    acc_per_fold =[]
    loss_per_fold=[]
        
    for train_index, test_index in kfold.split(inputs, targets):

        X_train, X_test = inputs[train_index], inputs[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        # store checkpoint with fold number and epoch info
        checkpoint=os.path.join(checkpoint_path,f"fine_fold_{fold_no}_"+"cp-best.weights.h5")
        print("checkpoint:",checkpoint)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                    save_weights_only=True,
                                                    monitor="loss",
                                                    verbose=1,
                                                    save_freq="epoch",
                                                    save_best_only=True)
    
        model.compile(optimizer=Adam(learning_rate=LEARNING_RATE/10),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(name='ft_loss'),
                weighted_metrics=[SparseCategoricalAccuracy(name='ft_accuracy')])

        
        history = model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_test,y_test),
                    epochs=total_epochs,
                    batch_size=BATCH_SIZE,    
                    # class_weight=weights,
                    callbacks=[model_callbacks,cp_callback])
        
        # Generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=0)
        print(scores)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        
        # Increase fold number
        fold_no = fold_no + 1
        
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
        print('------------------------------------------------------------------------')
        print('Average scores for all folds:')
        print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
        print(f'> Loss: {np.mean(loss_per_fold)}')
        print('------------------------------------------------------------------------')
    # get index of best accuracy , +1 since fold number starts with 1
    best_accuracy = np.argmax(acc_per_fold)
    print(f"fold {best_accuracy+1} has best accuracy with {acc_per_fold[best_accuracy]}")
    # get index of best loss
    best_loss = np.argmin(loss_per_fold)
    print(f"fold {best_loss+1} has best loss value with {loss_per_fold[best_loss]}")
    
    # load best weights 
    best_weights_path = os.path.join(checkpoint_path,f"fine_fold_{best_accuracy+1}_"+"cp-best.weights.h5")
    print("best weights in:",best_weights_path)
    model.load_weights(best_weights_path)
    scores = model.evaluate(val_inputs, val_targets, verbose=0)
    print('------------------------------------------------------------------------')
    print('Top Score:')
    print(scores)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    print('------------------------------------------------------------------------')


    
    # path for saving
    keras_save_path= os.path.join(save_path,"keras_fine_models",formatted_date,model_name,"model.keras")
    keras_save_dir = os.path.dirname(keras_save_path)
    if not os.path.exists(keras_save_dir):
        os.makedirs(keras_save_dir)
    # save keras model
    model.save(keras_save_path)
    print('')
    print('Initial training done. \n', flush=True)
    print('Keras model saved to .', keras_save_path, flush=True)