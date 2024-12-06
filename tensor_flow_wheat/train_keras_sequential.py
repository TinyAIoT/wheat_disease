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

    if weights_path is None:
        logging.info("Not using pretrained weights. Training from scratch.")
        train_base_model = True
           
    if include_top == True and weights_path=="imagenet":
        logging.info("Setting number of classes to 1000 because include_top = True and weights = 'imagenet' ")
        num_classes = 1000
        train_base_model = False
        
    base_model.trainable = train_base_model

    # batch size
    BATCH_SIZE =  args.batch_size
    # get dataset from directory
    set_class_names = ["brown_rust","healthy","mildew","septoria","yellow_rust"] 
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

    # labels for getting class_names
    y_train =[]
    for images, labels  in train_ds.unbatch():
        y_train.append(labels.numpy())

    # epochs and learning rate
    EPOCHS =  args.epochs
    LEARNING_RATE =  args.learning_rate
    logging.info(f"Using learning-rate: {LEARNING_RATE} ")
    #print(train_ds)
    
    # image augmentation
    train_ds = train_ds.map(mf.augment_image, num_parallel_calls=AUTOTUNE)
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # create model with prev. defined function
    model=mf.create_model(INPUT_SHAPE,base_model,LEARNING_RATE,num_classes,False,model_name)

    # train_sample_count ?
    print("card",train_ds.cardinality().numpy())

    train_sample_count = train_ds.cardinality().numpy()
    train_sample_count = math.ceil(train_sample_count) 
    # callbacks
    model_callbacks =[]
    # BatchLoggerCallback
    #model_callbacks.append(BatchLoggerCallback(BATCH_SIZE, train_sample_count, epochs=EPOCHS))
    
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
        patience=patience,               # Stop after 5 epochs without improvement
        verbose=1,                 # Print messages
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity.
    ))
    # learning rate printer
    lr_printer=LearningRatePrinter()
    model_callbacks.append(lr_printer)
    # learning rate scheduler
    lr_scheduler = get_lr_scheduler()
    model_callbacks.append(lr_scheduler)

    # Get the current date and time for path
    now = datetime.now()
    formatted_date = now.strftime("%d.%m.%Y")  
    print("Formatted date:", formatted_date)
    
    # Create a callback that saves the model's weights
    save_path =args.save_path
    checkpoint_path = os.path.join(save_path,"training_checkpoints",formatted_date,model_name,"cp-{epoch:04d}.weights.h5")
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

    model.summary(show_trainable=True)
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
    #print(f"checkpoints in {checkpoint_dir} ",os.listdir(checkpoint_dir))
    
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
    
    # keras model is created
    
    # Test model if true 
    if test_model:
        y_test = np.concatenate([y for x, y in test_ds], axis=0)
        X_test = np.concatenate([x for x, y in test_ds], axis=0)
        keras_model = model #tf.keras.models.load_model(keras_save_path)       
        logging.info("Evaluating on test data")
        results = keras_model.evaluate(X_test, y_test, return_dict=True)
        logging.info(f"test results for {model_name}: \n")
        print(results)
        #Predict
        y_prediction = keras_model.predict(X_test,batch_size=BATCH_SIZE)
        y_prediction = np.argmax (y_prediction,axis=1)

        #Create confusion matrix and normalizes it over predicted (columns)
        confusion_m = confusion_matrix(y_test, y_prediction,normalize="pred")
        logging.info(f"Confusion Matrix for {model_name}: \n")
        logging.info(confusion_m)