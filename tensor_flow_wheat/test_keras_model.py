import tensorflow as tf
from tensorflow import math
import argparse
from keras import utils
import numpy as np
from sklearn.metrics import confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--keras_savepath', type=str, required=True, help='path to folder where keras model is saved')
    parser.add_argument('--model_name', type=str, required=True, help='name of model')
    parser.add_argument('--testdata_path', type=str, required=True, help='path to folder where test data is stored')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--print_summary', type=bool, required=False, help='if True, will print model summary. False on default')
    parser.add_argument('--image_dim', type=int, required=False, help='image dimension x. if set input shape will be (x,x,3)')

    # parse arguments
    args = parser.parse_args()
    keras_save_path =args.keras_savepath
    # image dim
    IMAGE_HEIGHT= args.image_dim or 160
    IMAGE_WIDTH= args.image_dim or 160
    # batch size
    BATCH_SIZE = args.batch_size
    # get test data as dataset
    test_data_path =args.testdata_path
    test_ds = utils.image_dataset_from_directory(
        test_data_path,
        "inferred",
        "int",
        seed=1337,
        image_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
        batch_size=BATCH_SIZE,  
    )
    print(test_ds)

    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    
    X_test = np.concatenate([x for x, y in test_ds], axis=0)
    #print(X_test[0])
    
    keras_model = tf.keras.models.load_model(keras_save_path)
    print_summary = args.print_summary or False
    model_name = args.model_name
    if print_summary:
        print(f"\n Summary for {model_name}:\n",keras_model.summary())
        
    print("Evaluate on test data")
    results = keras_model.evaluate(X_test, y_test, return_dict=True)
    print(f"test results for {model_name}: \n")
    print(results)
    #Predict
    y_prediction = keras_model.predict(X_test,batch_size=args.batch_size)
    y_prediction = np.argmax (y_prediction,axis=1)

    #Create confusion matrix and normalizes it over predicted (columns)
    confusion_m = confusion_matrix(y_test, y_prediction,normalize="pred")
    print(f"Confusion Matrix for {model_name}: \n")
    print(confusion_m)
