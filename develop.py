import streamlit as st

from ml_models import predict_captions
import cv2

import numpy as np

st.title('Image Caption Predictor')

st.header('Input Image')

img=st.file_uploader('upload a image')

#st.text(img.shape)
if img:

    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.resize(opencv_image,(224,224),interpolation=cv2.INTER_AREA)
    opencv_image = opencv_image/255.0
    #plt.imsave('xxx.png',opencv_image)

    #img = cv2.imread('xxx.png', cv2.IMREAD_UNCHANGED)
    #print(img.shape)
    text=predict_captions(opencv_image)

    #im=cv2.resize(im,dsize=(1200,720))
    st.text('Input Image')
    st.image(img)
    st.title('Predicted Caption')
    st.header(text)