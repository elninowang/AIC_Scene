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

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import *

top3_acc = functools.partial(tf.contrib.keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
model = load_model("models/mixed/1-optimizer,adam-dropout,0.5-lr,0.0001-tune_layer,(140,173,96)__000- Acc:0.939.h5", custom_objects={'top3_acc': top3_acc})
print("load successed")

import json


def gen_test_result(model, model_image_size, json_name):
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory(dir + "test/", model_image_size, shuffle=False,
                                             batch_size=batch_size, class_mode=None)

    y_pred = model.predict_generator(test_generator, steps=test_generator.samples // batch_size + 1, verbose=1)
    print("y_pred shape {}".format(y_pred.shape))
    print(y_pred[0])

    l = list()
    for i, fname in enumerate(test_generator.filenames):
        name = fname[fname.rfind('/') + 1:]
        d = dict()
        d["image_id"] = name
        d["label_id"] = y_pred[i].argsort()[-3:][::-1].tolist()
        l.append(d)

    json.dump(l, open(json_name, "w"))
    print("json saved")


print("done")

gen_test_result(model,  model_image_size, 'json/mixed-imagenet-finetune-pred.json')