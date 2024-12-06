import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
#from keras import layers, metrics
from tensorflow.keras.layers import (
    Dense, InputLayer, Dropout, Flatten, Reshape,RandomFlip,RandomRotation,Rescaling)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.applications import MobileNetV3Small,MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3,EfficientNetV2B1
import os
import logging

def create_model(input_shape,base_model,learning_rate:float,num_classes:int, plot_model:bool,model_name:str):
    model = Sequential(name=model_name)
    model.add(InputLayer(input_shape=input_shape, name='x_input'))
    # rescaling, rotation and flip
    model.add(Rescaling(1./127.5,offset=-1))
    model.add(RandomRotation(0.2))
    model.add(RandomFlip("horizontal_and_vertical"))
    # Don't include the base model's top layers
    last_layer_index = -3
    model.add(Model(inputs=base_model.inputs, outputs=base_model.layers[last_layer_index].output)) 
    model.add(Reshape((-1, model.layers[-1].output.shape[3])))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    
    # compile
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                weighted_metrics=[SparseCategoricalAccuracy()])
                #,Recall(name="recall"),,Precision(name="precision")])
                #metrics=['accuracy'])
    # plot model to png
    if plot_model:
        path = os.path.join(os.path.dirname(__file__),'keras_models',model_name)
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

def get_base_model(base_model_name:str,input_shape,weights_path,include_top:bool,num_classes:int,dropout_rate,alpha,pooling):
    if base_model_name == "mobile_net_v2":
        logging.info("mobile_net_v2 as base model ... \n")
        # MobileNetV2
        # If using `weights` as `"imagenet"` with `include_top` as true, `classes` should be 1000.
        base_model = MobileNetV2(
            input_shape = input_shape,
            alpha=alpha,
            pooling=pooling,
            weights = weights_path,
            include_top = include_top,
            classes=num_classes, # only to be specified if include_top is True, and if no weights argument is specified
            #pooling="avg" 
        )      
    elif base_model_name == "mobile_net_v3_s":
        logging.info("mobile_net_v3_small as base model ... \n")
        # MobileNetV3 small
        base_model = MobileNetV3Small(
            input_shape = input_shape,
            alpha=alpha,
            pooling=pooling,
            dropout_rate=dropout_rate,
            weights = weights_path,
            include_top = include_top,
            classes=num_classes, # only to be specified if include_top is True, and if no weights argument is specified
        )
    elif base_model_name == "mobile_net_v3_l":
        logging.info("mobile_net_v3_large as base model ... \n")
        # MobileNetV3 large
        base_model = MobileNetV3Large(
            input_shape = input_shape,
            alpha=alpha,
            pooling=pooling,
            dropout_rate=dropout_rate,
            weights = weights_path,
            include_top = include_top,
            classes=num_classes, # only to be specified if include_top is True, and if no weights argument is specified
        )
    elif base_model_name == "effnet_v2_b3":
        logging.info("EfficientNetV2B3 as base model ... \n")
        # EfficientNetV2B3
        base_model = EfficientNetV2B3(
            input_shape = input_shape,
            weights = weights_path,
            pooling=pooling,
            include_top = include_top,
            classes=num_classes, 
        )
    elif base_model_name == "effnet_v2_b1":
        logging.info("EfficientNetV2B3 as base model ... \n")
        # EfficientNetV2B3
        base_model = EfficientNetV2B1(
            input_shape = input_shape,
            weights = weights_path,
            pooling=pooling,
            include_top = include_top,
            classes=num_classes, 
        )
    else:
        raise ValueError(f"No base model with name {base_model_name}. Try one of the following names : [effnet_v2_b1,effnet_v2_b3,mobile_net_v3_l,mobile_net_v3_s,mobile_net_v2]")
        
    return base_model

# Implements the data augmentation policy
def augment_image(image, label):
    
    # Vary the brightness of the image
    image = tf.image.random_brightness(image, max_delta=0.2)

    return image, label

def zen3_config():
    #config = tf.ConfigProto()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        #sess = tf.Session(config=config)
        sess = tf.compat.v1.Session(config=config)
        #os.system("export TF_GPU_ALLOCATOR=cuda_malloc_async")
        #TF_GPU_ALLOCATOR=cuda_malloc_async
        return sess