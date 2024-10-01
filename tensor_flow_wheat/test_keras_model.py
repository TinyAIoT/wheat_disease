import tensorflow as tf
import argparse
from keras import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--keras_savepath', type=str, required=True, help='path to folder where keras model is saved')
    parser.add_argument('--testdata_path', type=str, required=True, help='path to folder where test data is stored')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')

    # parse arguments
    args = parser.parse_args()
    keras_save_path =args.keras_savepath
    # image dim
    IMAGE_HEIGHT= 160
    IMAGE_WIDTH= 160
    # batch size
    BATCH_SIZE = args.batch_size
    # get test data
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

    keras_model = tf.keras.models.load_model(keras_save_path)
    print("Evaluate on test data")
    results = keras_model.evaluate(test_ds, batch_size=BATCH_SIZE, return_dict=True)
    print(results)
