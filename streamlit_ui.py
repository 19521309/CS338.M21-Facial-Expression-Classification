"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# set title of app
st.title("Simple Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")

from keras.models import load_model
model = load_model("ResNet152.h5")
labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

def predict (file_up):
    original_image = cv2.imread(file_up)
    images = np.array(original_image)
    img_resized = cv2.resize(images,(48, 48),-1)
    img_new = np.expand_dims(img_resized, 0)
    rgb_test_img = np.repeat(img_new[..., np.newaxis], 3, -1)
    pred = model.predict(rgb_test_img)
    output_class=labels[np.argmax(pred)]    
    result = ("The predicted class is", output_class)
    return result
    

if file_up is not None:
    # display image that user uploaded

    image = Image.open(file_up)
   
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    images = np.array(image)
    img_resized = cv2.resize(images,(48, 48))
    img_new = np.expand_dims(img_resized, 0)
    pred = model.predict(img_new)
    output_class=labels[np.argmax(pred)]    
    st.write("The predicted class is", output_class)
    
    
