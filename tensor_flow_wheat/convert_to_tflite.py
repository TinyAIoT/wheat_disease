from tflite_functions import convert_tflite_model,save_tflite_model,test_tflite
import tensorflow as tf
import os
import argparse


if __name__ == "__main__":
    # arguments for tf lite converting
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--keras_savepath', type=str, required=True, help='path where keras model is stored.')
    parser.add_argument('--tf_lite_modelname', type=str, required=True, help='name which is used to save tf lite model.')

    
    # parse arguments
    args = parser.parse_args()
    keras_save_path=args.keras_savepath
    model_name =args.tf_lite_modelname
    

# load model after saving
    keras_model = tf.keras.models.load_model(keras_save_path)

    # convert tflite
    print("model to be converted:",keras_model)
    tflite_model = convert_tflite_model(keras_model)

    # save tflite
    tflite_save_path = os.path.join(os.path.dirname(__file__),'tf_lite_models',model_name)
    save_tflite_model(tflite_model, tflite_save_path, 'model.tflite')