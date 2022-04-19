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

import json
from types import SimpleNamespace

config = sys.argv[1]
cfg = json.load(open(config,"r"), object_hook=lambda d: SimpleNamespace(**d))
cfg.config=os.path.basename(config)

    
    
if __name__=="__main__":
    json_cfg = json.dumps({x:cfg.__dict__[x] for x in cfg.__dict__ if not x.startswith("_")})
    print("JSONCFG: %s" % (json_cfg))
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
    
    use_metrics=[]
    print(model.summary())
    model.compile(optimizer=optimizer, loss = loss_fn, metrics=use_metrics)
    # data

        
    
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1 / 255.0)
    train_generator = train_datagen.flow_from_directory(
    directory= "./%s/train/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    valid_generator = train_datagen.flow_from_directory(
        directory="./%s/val/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
    )
    test_generator = train_datagen.flow_from_directory(
        directory="./%s/test/" %(cfg.dataset),
        target_size=(cfg.img_input_shape[0],cfg.img_input_shape[1]),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
    )
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

    # Sometimes, your weights will go NaN (e.g., too large LR)
    # and it is not automatic that the job ends then. But we want to.
    
    callbacks = [ keras.callbacks.TerminateOnNaN() ]

    
    model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                        epochs=cfg.epochs,
                        callbacks = callbacks
    )
    model.save("%s.h5" % (cfg.config))
    # Evaluate on Testset
    use_metrics = [keras.metrics.Precision(),keras.metrics.Recall()]

    model.compile(optimizer=optimizer, loss = loss_fn, metrics=use_metrics)
    
    STEPS_TEST=test_generator.n//test_generator.batch_size
    results = json.dumps(dict(zip(model.metrics_names, model.evaluate(test_generator, steps=STEPS_TEST))))
    print("TEST:%s" %(results))
