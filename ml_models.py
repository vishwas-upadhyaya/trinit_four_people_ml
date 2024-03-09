import pandas as pd
import numpy as np
import os
import cv2
import csv
import random
import math
from shapely.geometry import Polygon
from itertools import compress
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, BatchNormalization, Concatenate, Input, Dense, Flatten, \
    Activation
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

embedding_matrix = np.load('embedding_matrix.npy')


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)
    
    
model = tf.lite.Interpreter(model_path="model.tflite")
model.allocate_tensors()


input_details_model = model.get_input_details()
output_details_model = model.get_output_details()

    
def predict_captions(image):
    start_word = ["sos"]
    c=0
    while True:
        par_caps = [tokenizer1.word_index[i] for i in start_word]
#         print(np.array(par_caps).shape)
        par_caps = pad_sequences([par_caps], maxlen=20, padding='post')
#         print(np.array(par_caps).shape)
        par_caps = [embedding_matrix[i,:] for i in par_caps]
#         print(np.array(par_caps).shape)
        model.set_tensor(input_details_model[0]['index'], np.expand_dims(image,axis=0).astype(np.float32))
        model.set_tensor(input_details_model[1]['index'], np.array(par_caps).astype(np.float32))
        model.invoke()
        preds = model.get_tensor(output_details_model[0]['index'])
#         print(np.sum(preds[0][c]))
#         if c!=1:
        word_pred = tokenizer1.index_word[np.argmax(preds[0][c])]
#         else:
#             word_pred = 'a'
        start_word.append(word_pred)
        c+=1
        
        if word_pred == "eos" or len(start_word) > 20:
            break
            
    return ' '.join(start_word[1:-1])

# id_no = 78
# # 'archive/Images/'+X_test['image'].values[id_no]
# img = cv2.imread('archive/images/'+X_test['image'].values[71])
# # img = cv2.imread('test/test11.jpg')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# img = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
# img = img/255.0
# Argmax_Search = predict_captions(img)


