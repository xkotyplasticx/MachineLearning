#pip install opencv-python
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

import cv2
from sklearn.preprocessing import StandardScaler

st.title ('Recognize a number')

 #if image is not None:
clf = joblib.load('knn_model.pkl')
   

def processImage(input):
        
    # Read input
    data_in = input.getvalue()

    # Decode to grayscale and use Thresholding (Turns image BnW)
    # Convert the image to an array and resize
    decode = cv2.imdecode(np.frombuffer(data_in, np.uint8), cv2.IMREAD_GRAYSCALE)
    _, thresholded = cv2.threshold(decode, threshVal, 255, cv2.THRESH_BINARY_INV)      # first variable _ is a dummy because return values are two
    imgArr = thresholded/255
    resized = cv2.resize(imgArr, (28, 28), interpolation=cv2.INTER_AREA)
    
    # standardize the image
    scaleImg = StandardScaler()
    fix_the_pic = scaleImg.fit_transform(resized.reshape(-1, 1)).reshape(resized.shape)

    # Display the reformatted image
    st.write("Reformatted image: ")
    st.image(fix_the_pic, width=128 ,output_format="auto", clamp=True)

    if st.button('Make Prediction'):
       prediction(fix_the_pic)

    
def prediction(inputImg):
     
    # Model prediction
    flat = inputImg.flatten().reshape(1, -1)
    st.write("Predicted digit: ", clf.predict(flat))

# ------------------------------------------------

mode = st.toggle('Uppload | Take a picture of a digit', value = True)    
if mode:

    buffer = st.camera_input("Take a picture with your webcam of a handwritten single digit!")
    threshVal = st.slider('Threshold value', 0, 254, 127)

    if buffer is not None:
        processImage(buffer)
        #st.balloons()
else:
    upload = st.file_uploader("Upload your picture of a single digit here:", type=['png', 'jpeg', 'jpg'])
    threshVal = st.slider('Threshold value', 0, 254, 127)

    if upload is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.write("Uploaded image: ")
            st.image(upload, width=128)
        with col2:
            processImage(upload)


