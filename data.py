import json
import os
import shutil
import cv2

dir = "/ext/Data/aichallenger/scene"

first_train_dir = os.path.join(dir, "ai_challenger_scene_train_20170904")
first_valid_dir = os.path.join(dir, "ai_challenger_scene_validation_20170908")
first_train_img_dir = os.path.join(first_train_dir, "scene_train_images_20170904")
first_valid_img_dir = os.path.join(first_valid_dir, "scene_validation_images_20170908")
first_train_json = os.path.join(first_train_dir, "scene_train_annotations_20170904.json")
first_valid_json = os.path.join(first_valid_dir, "scene_validation_annotations_20170908.json")

second_train_dir = os.path.join(dir,"train")
second_valid_dir = os.path.join(dir,"valid")

def createimages(json_file, first_dir, second_dir):
    json_obj = json.load(open(json_file, "r"))
    count = 0
    for obj in json_obj:
        target_dir = os.path.join(second_dir, obj["label_id"])
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        #shutil.copy(os.path.join(first_dir, obj["image_id"]), target_dir)
        img = cv2.imread(os.path.join(first_dir, obj["image_id"]))
        flip_img = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(target_dir, "_" + obj["image_id"]), flip_img)
        count += 1
        if count % 2000 == 0:
            print("copy %d images" % count)

#createimages(first_valid_json, first_valid_img_dir, second_valid_dir)
createimages(first_train_json, first_train_img_dir, second_train_dir)


