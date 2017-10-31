import os
import cv2
import glob
import numpy as np
import pandas as pd

from keras.models import *
from keras.optimizers import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.callbacks import *
from keras.layers import *
import itertools

import tensorflow as tf
import functools

dir = "/ext/Data/aichallenger/scene/"

model_image_size = (299, 299)
batch_size = 32

classdf = pd.read_csv("scene_classes.csv")
for i in range(80):
    if i % 10 == 9:
        print(classdf.loc[i]["chinese"])

train_gen = ImageDataGenerator(
#     featurewise_std_normalization=True,
#    samplewise_std_normalization=False,
#     rotation_range=10.,
#     width_shift_range=0.05,
#     height_shift_range=0.05,
#     shear_range=0.1,
#     zoom_range=0.1,
)
gen = ImageDataGenerator(
#     featurewise_std_normalization=True,
#     samplewise_std_normalization=False,
)

classes = []
for i in range(80):
    classes.append(str(i))
train_generator = train_gen.flow_from_directory(os.path.join(dir, 'train'),  model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical", classes=classes)
print("subdior to train type {}".format(train_generator.class_indices))
valid_generator = gen.flow_from_directory(os.path.join(dir, 'valid'),  model_image_size, shuffle=True, batch_size=batch_size, class_mode="categorical", classes=classes)
print("subdior to valid type {}".format(valid_generator.class_indices))

top3_acc = functools.partial(tf.contrib.keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'

def make_model(optimizer, dropout, lr, tune_layer_resnet50, tune_layer_inceptionV3, tune_layer_xception):
    input_tensor = Input((*model_image_size, 3))
    x = input_tensor
    inception_v3_x = Lambda(inception_v3.preprocess_input)(x)
    xception_x = Lambda(xception.preprocess_input)(x)

    # resnet50_model = ResNet50(input_tensor=x, weights='imagenet', include_top=False)
    # resnet50_model_output = GlobalAveragePooling2D()(resnet50_model.output)
    # for i in range(tune_layer_resnet50):
    #     resnet50_model.layers[i].trainable = False

    inceptionV3_model = InceptionV3(input_tensor=inception_v3_x, weights='imagenet', include_top=False)
    inceptionV3_model_output = GlobalAveragePooling2D()(inceptionV3_model.output)
    for i in range(tune_layer_inceptionV3):
        inceptionV3_model.layers[i].trainable = False

    xception_model = Xception(input_tensor=xception_x, weights='imagenet', include_top=False)
    xception_model_output = GlobalAveragePooling2D()(xception_model.output)
    for i in range(tune_layer_xception):
        xception_model.layers[i].trainable = False

    #x = Concatenate(axis=-1)([resnet50_model_output, inceptionV3_model_output, xception_model_output])
    #x = resnet50_model_output
    x = Concatenate(axis=-1)([inceptionV3_model_output, xception_model_output])
    x = Dropout(dropout)(x)
    x = Dense(80, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)

    print("total layer count {}".format(len(model.layers)))

    if optimizer == "adam":
        optimizer_class = Adam(lr=lr)
    elif optimizer == "rmsprop":
        optimizer_class = RMSprop(lr=lr)

    model.compile(optimizer=optimizer_class, loss='categorical_crossentropy', metrics=[top3_acc])
    return model

print("train_generator.samples = {}".format(train_generator.samples))
print("valid_generator.samples = {}".format(valid_generator.samples))
steps_train_sample = train_generator.samples // batch_size + 1
steps_valid_sample = valid_generator.samples // batch_size + 1

if not os.path.exists("models/mixed"):
    os.mkdir("models/mixed")

optimizers = ['adam']
dropouts = [0.5, 0.25]
lrs = [0.0001,  0.001]
#resnet50_tune_layers = [140, 162, 172]
resnet50_tune_layers = [140]
inceptionV3_tune_layers = [173,  213,  253]
xception_tune_layers = [96, 116,  126]

parameters = itertools.product(optimizers, dropouts, lrs, resnet50_tune_layers, inceptionV3_tune_layers, xception_tune_layers)

skip_count = 11
count = 0
for p in parameters:
    count += 1
    name = "{}-optimizer,{}-dropout,{}-lr,{}-tune_layer,({},{},{})".format(count, p[0],p[1],p[2],p[3],p[4],p[5])
    if count <= skip_count:
        print("skip " +  name);
        continue
    final_filepath = "models/mixed/" + name + ".h5"
    best_filepath="models/mixed/" + name + "__{epoch:03d}- Acc:{val_top3_acc:.3f}.h5"
    print()
    print(name)
    callbacks = [EarlyStopping(monitor='val_top3_acc',patience=4),ModelCheckpoint(best_filepath, monitor='val_top3_acc',save_best_only=True)]
    model = make_model(p[0],p[1],p[2],p[3],p[4],p[5])
    model.fit_generator(train_generator,
                                                  steps_per_epoch=steps_train_sample,
                                                  validation_data=valid_generator,
                                                  validation_steps=steps_valid_sample,
                                                  epochs=500,
                                                  callbacks=callbacks)
    model.save(final_filepath)
print("model saved!")

