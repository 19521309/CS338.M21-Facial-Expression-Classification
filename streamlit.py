"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# set title of app
st.title("Facial Expression Regconition")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")

from keras.models import load_model
model = load_model("ResNet50.h5")

    

if file_up is not None:
    # display image that user uploaded
    #setting image resizing parameters
    image = Image.open(file_up)
   
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    
     #loading image
    images = np.array(image)
    test_image = images
    
    x_min = test_image.shape[0]/15
    y_min = test_image.shape[1]/15

    box_thickness = 2
    line_thickness = 1
    if (test_image.shape[0] > 500 and test_image.shape[0] < 1000):
        box_thickness = 4
        line_thickness = 2
    elif (test_image.shape[0] > 1000 and test_image.shape[0] < 1500):
        box_thickness = 6
        line_thickness = 3
    elif (test_image.shape[0] > 1500):
        box_thickness = 10
        line_thickness = 5
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

   
    gray = cv2.cvtColor(images,cv2.COLOR_RGB2GRAY)
    face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray, scaleFactor  = 1.05,minNeighbors =  6, minSize = (int(x_min), int(y_min)))

    emotion = ""
    #detecting faces
    for (x, y, w, h) in faces:
        image = Image.open(file_up)
        roi_gray = gray[y:y + h, x:x + w]
        print(faces)
        cropped_img = np.expand_dims(cv2.resize(roi_gray, (48, 48)), 0)
        rgb_cropped_img = np.repeat(cropped_img[..., np.newaxis], 3, -1)
        rgb_cropped_img = rgb_cropped_img/255.
        cv2.rectangle(images, (x, y), (x + w, y + h), (255, 0, 0), box_thickness)
        #predicting the emotion
        pred= model.predict(rgb_cropped_img)
        cv2.putText(images, labels[(np.argmax(pred))] + ":" + str(np.amax(pred)) , (x, y), cv2.FONT_HERSHEY_SIMPLEX, line_thickness, (255, 255, 0), line_thickness, cv2.LINE_AA)
        emotion = labels[(np.argmax(pred))]
        print(pred)
        print("Emotion: "+labels[(np.argmax(pred))])
    fig = plt.figure(figsize=(20,10))
    plt.title('Predict')
    plt.imshow(images)
   
    st.pyplot(fig) 
    
    
