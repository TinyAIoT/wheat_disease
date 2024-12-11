import math
import os
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras import utils
import numpy as np
from sklearn.utils import class_weight
# additional callbacks and tflite functions
import argparse
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from datetime import datetime
import logging
from tf_costum_utils import model_functions as mf
from tf_costum_utils.callbacks import (BatchLoggerCallback,
LearningRatePrinter,get_callback_list,get_lr_scheduler)

if __name__ == "__main__":
    # arguments for model training
    parser = argparse.ArgumentParser(description='Train Sequential model with tensorflow')
    parser.add_argument('--data_folder', type=str, required=True, help='path to folder with validation and training data')
    parser.add_argument('--base_model', type=str, required=False, help='Set base model. e.g mobile_net_v2 or mobile_net_3')
    parser.add_argument('--include_top',default=False, type=bool, required=False, help='True if top layers of base model should be included')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes')
    parser.add_argument('--dropout', type=float, required=True, help='dropout')
    parser.add_argument('--save_path', type=str, required=True, help='saving path for keras models and checkpoints')
    parser.add_argument('--pt_weights', type=str,default=None, required=False, help='pretrained weights for base model. If not available, will download weights')
    parser.add_argument('--model_name', type=str, required=True, help='name of model. Also defines saving path')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, required=True, help='learning rate')
    parser.add_argument('--patience', type=int, required=False, help='patience for early stopping')
    parser.add_argument('--min_delta', type=float, required=False, help='mind_delta for early stopping')
    parser.add_argument('--test_model', type=bool, required=False, help='if true val dataset will be split again and model will be tested ')
    parser.add_argument('--image_dim', type=int, required=False, help='image dimension x. if set input shape will be (x,x,3)')
    parser.add_argument('--zen3', type=bool, required=False, help='Set True, if you are using zen3 for config')

    logging.basicConfig(level = logging.INFO,format='%(asctime)s - %(levelname)s: %(message)s',datefmt='%H:%M:%S')
    tf.get_logger().setLevel(logging.ERROR)
    # parse arguments
    args = parser.parse_args()
    
    zen3=args.zen3 or False
    if zen3:
       sess = mf.zen3_config()
       
    # epochs and learning rate
    EPOCHS =  args.epochs
    LEARNING_RATE =  args.learning_rate
    lr_list=mf.generate_floats_around(LEARNING_RATE,6,0.0002)
    logging.info(f"Using learning-rate: {LEARNING_RATE} ")
    logging.info(f"Using multiple learning rates: {lr_list} ")
       
    # Image dimensions
    # path_name
    model_name =args.model_name
    logging.info(f"Creating model {model_name} ... \n")
    data_folder = args.data_folder
    logging.info(f"Using data from {data_folder} \n")
    IMAGE_HEIGHT= args.image_dim or 320   
    IMAGE_WIDTH= args.image_dim or 320
    # Input shape
    INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    
    base_model_name=args.base_model
    weights_path = args.pt_weights or None
    include_top=args.include_top
    num_classes=args.num_classes
    dropout=args.dropout
    alpha=1.0
    
    base_model= mf.get_base_model(base_model_name,INPUT_SHAPE,
                                weights_path,include_top,num_classes,dropout,alpha,None)

    if weights_path is None and include_top==True :
        logging.info("Not using pretrained weights. Training from scratch.")
        logging.info("Including top layers of base model")
        num_classes = args.num_classes
        logging.info(f"num_classes: {num_classes}")
           
    else:
        num_classes = 1000
        
        
    base_model.trainable = False

    # batch size
    BATCH_SIZE =  args.batch_size
    # get dataset from directory
    #set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"] 
    #"int": means that the labels are encoded as integers (e.g. for sparse_categorical_crossentropy loss).
    train_ds, val_ds = utils.image_dataset_from_directory(
        data_folder,
        "inferred", #labels are generated from the directory structure
        "int",
        validation_split=0.3,
        subset="both",
        seed=1337,
        image_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
        batch_size=BATCH_SIZE,  
    )
    # Autotune
    AUTOTUNE = tf.data.AUTOTUNE
    # if test_model == true val datset will be split to val and test
    test_model = args.test_model or False
    if test_model:
        
        val_ds,test_ds=tf.keras.utils.split_dataset(
        val_ds, left_size=0.6, right_size=0.4, shuffle=True, seed=32)
        print("test_ds Cardinality: ",test_ds.cardinality().numpy())
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
        print("test_ds Cardinality: ",test_ds.cardinality().numpy())
        targets_test = np.concatenate([y for x, y in test_ds], axis=0)
        inputs_test = np.concatenate([x for x, y in test_ds], axis=0)

    # labels for getting class_names
    y_train =[]
    for images, labels  in train_ds.unbatch():
        y_train.append(labels.numpy())

    
    #print(train_ds)
    
    # image augmentation
    train_ds = train_ds.map(mf.augment_image, num_parallel_calls=AUTOTUNE)
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # create model with prev. defined function
    model=mf.create_model(INPUT_SHAPE,base_model,num_classes,model_name)
    
    # callbacks
    model_callbacks =[]
    # apply early stopping
    min_d = args.min_delta or 0.001
    patience=args.patience or 10
    logging.info(f"min_delta for early stopping: {min_d} \n")
    logging.info(f"patience for early stopping: {patience} \n")
    # append early stopping
    model_callbacks.append(EarlyStopping(
        monitor='val_sparse_categorical_accuracy',    # Monitor validation accuracy
        min_delta=min_d ,           # Minimum change to qualify as an improvement
        patience=patience,               # Stop after 5 epochs without improvement
        verbose=1,
        start_from_epoch=20,# Print messages
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    ))
    
    # Get the current date and time for path
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y")  
    print("Formatted date:", formatted_date)
    
    # Create a callback that saves the model's weights
    save_path =args.save_path
    checkpoint_path = os.path.join(save_path,"training_checkpoints",formatted_date,model_name)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    print("checkpoint_path",checkpoint_path)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.summary(show_trainable=True)
    # class weights
    print("np.unique(y_train):",np.unique(y_train))
    weights = class_weight.compute_class_weight('balanced',
                                                    classes= np.unique(y_train),
                                                    y= y_train)
    weights = {i : weights[i] for i in range(5)}
    print("classweights",weights ,"\n")
    print("----------")
    results={}
    for lr in lr_list:
        # compile
        base_model.trainable = False
        if lr <= 0:
                logging.error(f"Learning rate is too small ! : {lr} \n Continue ...")
                continue
        logging.info(f"Training with Learning rate {lr} ... \n")
        checkpoint=os.path.join(checkpoint_path,f"_lr{lr}_"+"cp-best.weights.h5")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint,
                                                    save_weights_only=True,
                                                    monitor="val_loss",
                                                    verbose=1,
                                                    save_freq="epoch",
                                                    save_best_only=True)
        
        model.compile(optimizer=Adam(learning_rate=lr),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    weighted_metrics=[SparseCategoricalAccuracy()])
        # fit model
        history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,    
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
                                validation_data=val_ds,
                                callbacks=[model_callbacks,cp_callback])
    # Generate generalization metrics
        scores = model.evaluate(inputs_test, targets_test, verbose=0)
        logging.info(f'Score for learning rate {lr}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        accuracy = scores[1] * 100
        loss = scores[0]
        results[lr] = {'accuracy': accuracy, 'loss': loss}
    
    # get best learning rate
    best_lr = None
    best_accuracy = float('-inf')
    for lr, metrics in results.items():
        logging.info(f"LR {lr} : {metrics} \n")
        if metrics['accuracy'] > best_accuracy:  # Use accuracy or loss as criteria
            best_accuracy = metrics['accuracy']
            best_lr = lr
            
    logging.info(f"learning rate {best_lr} has best accuracy with {results[best_lr]}")
    # load best weights 
    best_weights_path = os.path.join(checkpoint_path,f"_lr{best_lr}_"+"cp-best.weights.h5")
    logging.info(f"best weights in: {best_weights_path}")
    # load best weights
    model.load_weights(best_weights_path)
    
    
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
    # confusion matrix
    y_prediction = model.predict(inputs_test,batch_size=BATCH_SIZE)
    #Finds the index of the maximum probability (the most likely class)
    y_prediction = np.argmax (y_prediction,axis=1)

    #Create confusion matrix and normalizes it over predicted (columns)
    confusion_m = confusion_matrix(targets_test, y_prediction,normalize="pred")
    logging.info(f"Confusion Matrix for {model_name}: \n")
    logging.info(confusion_m)
    
    