import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

from numpy import dot
from numpy.linalg import norm
import os
import sys
import numpy as np

class cfg:
    img_input_shape= (64,64,3) # 224,224(224, 224)
    number_of_classes = 2
    lr1 = np.random.choice([0.01, 0.001, 0.0001])
    epochs=np.random.choice([10,25, 50,100])
    hidden_units = np.random.choice([32,64,128,256,512,1024])
    dropout = np.random.choice([0.1,0.2,0.5])


    
    
if __name__=="__main__":
    print("CFG: %s" % (str (cfg.__dict__)))

    conv = tf.keras.applications.vgg16.VGG16(weights='imagenet',
                                             include_top=False,
                                             input_tensor=None,
                                             pooling=None,
                                             input_shape=cfg.img_input_shape)
 
    new_inputs = tf.keras.layers.Input(shape=cfg.img_input_shape)
    x = conv(new_inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(cfg.hidden_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    new_outputs = tf.keras.layers.Dense(cfg.number_of_classes, activation='softmax')(x)

    model = tf.keras.Model(new_inputs, new_outputs)

    loss_fn = keras.losses.CategoricalCrossentropy() #from_logits=True
    optimizer = keras.optimizers.Adam(learning_rate = cfg.lr1)
    
    train_acc_metric = keras.metrics.Precision()
    print(model.summary())
    model.compile(optimizer=optimizer, loss = loss_fn, metrics=[train_acc_metric])
    # data

        
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0)
    train_generator = train_datagen.flow_from_directory(
    directory=r"./firepreview/train/",
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    valid_generator = train_datagen.flow_from_directory(
    directory=r"./firepreview/val/",
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=cfg.epochs
    )

    
