#encoding=utf-8
'''

'''
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
valid_input_path='C:/datas/scene_classification/input/validation/scene_validation_images_20170908/'
from models import get_incept3_model,senet,make_model
from sklearn.metrics import log_loss, accuracy_score,confusion_matrix,\
    classification_report
model_image_size = (299, 299)
# model_image_size = (64, 64)
pred_batch=64
valid_df=pd.read_csv("../input/validation/valid.csv")

def get_img(path, img_rows, img_cols):

    img = image.load_img(path, target_size=(img_rows, img_cols))
    img = image.img_to_array(img)
    return img

def load_valid(df,img_size):
    '''

    :param df:
    :param img_size:
    :return:
    '''
    X_valid=[]
    # img_ids = df['image_id'].tolist()
    # labels = df['label_id'].tolist()
    img_ids = df['image_id']
    labels = df['label_id'].tolist()
    print(len(img_ids), 'valid samples')
    from joblib import Parallel, delayed
    X_valid.extend(
        Parallel(n_jobs=4)(
            delayed(get_img)(valid_input_path + img_id, img_size, img_size) for img_id in img_ids)
    )
    # for i, img_id in tqdm(enumerate(img_ids)):
    #     img = get_img(valid_input_path + img_id, img_size, img_size)
    #     # img = transform(img)
    #     X_valid.append(img)

    return X_valid, labels

def get_argmax(valid_preds):
    y_preds=[]
    for e in valid_preds:
        y_preds.append(np.argmax(e))
    return y_preds

def get_confusion_matrix():
    gen = ImageDataGenerator()
    test_input_path = '../input/valid/'
    classes = [str(i) for i in range(80)]
    test_generator = gen.flow_from_directory(test_input_path, model_image_size, shuffle=False,
                                             classes=classes,
                                             batch_size=pred_batch, class_mode='categorical')
    y_test = test_generator.classes
    # X_valid, labels=load_valid(valid_df,299)
    import pickle
    # with open('cache/X_valid_299.pkl','wb') as fout:
    #     pickle.dump(X_valid,fout)
    # with open('cache/y_valid_299.pkl','wb') as fout:
    #     pickle.dump(labels,fout)
    # with open('cache/X_valid_299.pkl','rb') as fout:
    #     X_valid=pickle.load(fout)
    # with open('cache/y_valid_299.pkl','rb') as fout:
    #     labels=pickle.load(fout)
    # X_valid=np.array(X_valid)
    # X_valid=X_valid/255
    # X_valid=preprocess_input(X_valid)
    # model=get_incept3_model("adam",0.5,0.001,173)
    model=make_model('adam',0.5,0.001,0,173,96)
    # model=senet(img_size=64)
    # model.load_weights(filepath="weights/senet_01_1.524.h5")
    model.load_weights(filepath="weights/6-optimizer,adam-dropout,0.5-lr,0.0001-tune_layer,(140,213,126)__000- Acc_0.937.h5")
    y_pred = model.predict_generator(test_generator, steps=test_generator.samples // pred_batch + 1, verbose=1)
    # y_pred=model.predict(X_valid,batch_size=pred_batch)
    # y_test=labels

    print("y_test",y_test[0])
    print(y_pred.shape)
    print(np.argmax(y_pred[0]))
    valid_preds=get_argmax(y_pred)
    print(accuracy_score(y_test,valid_preds))
    print(classification_report(y_test, valid_preds))

    cf=confusion_matrix(y_test, valid_preds)
    print(cf)

if __name__=='__main__':
    get_confusion_matrix()