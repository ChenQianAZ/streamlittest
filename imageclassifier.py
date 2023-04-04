#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:03:58 2023

@author: kbxp708
"""

import base64
import io
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time


# Load the TFLite model
tflite_model_path = 'model/syrinergy_tflite_model.tflite'
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image):  
    # Resize and normalize the image 
    image = image.resize((300, 300), Image.ANTIALIAS).convert('RGB')
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0).astype(np.float32)  
    return image 

def classify_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = preprocess_image(img)
    input_data = img
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    return predictions

# Streamlit app layout
st.title("Image Classification Demo")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    t = time.process_time()
#do some stuff

    st.write("Classifying...")
    predictions = classify_image(uploaded_file.getvalue())
    class_names = ['before injection', 'after injection']  # Replace with your model's class names
    if predictions>0.5:
        result = class_names[1]
    else:
        result = class_names[0]
    elapsed_time = time.process_time() - t 
    st.write("Results:")
    st.write(result)
    st.write('inference used ',elapsed_time,' seconds')