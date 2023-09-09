import streamlit as st
#import SessionState
import tensorflow as tf
import requests
import json
import os
from utils import classes_and_models, predict_json
import numpy as np


# Setup environment credentias 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "dami-ml-projects-343dfacc9920.json"
PROJECT = "dami_ml_projects"
PROJECT_NUMBER = "368522488538"
REGION = "us-central1"

MODEL = classes_and_models["model_1"]["model_name"]
MODEL_ID = classes_and_models["model_1"]["model_id"]
ENDPOINT_ID = classes_and_models["model_1"]["endpoint_id"]
CLASSES = classes_and_models["model_1"]["classes"]



st.title("Welcome to Plant Pathology Kaggle challenge")
st.header("Identify plant leaves diseases")
st.header("A Deep learning app to help identify if an apple leaf is healthy, has multiple diseases, rust or scab")


# File uploader allows user to add their own image
upload_file = st.file_uploader(label="Please upload image of the apple leaf", 
                               type=["png", "jpg", "jpeg"])

### Function to load the model
@st.cache_resource
def load_model():
    #model = tf.keras.models.load_model("plant_class.keras")
    model = tf.keras.models.load_model("plant_class-2.h5")
    #print("Model loaded")
    
    return model

# Show spinner on loading
with st.spinner("Loading model..."):
    model = load_model()

# Classes and Preprocess the image
base_classes = ['healthy',
 'multiple_diseases',
 'rust',
 'scab'
 ]

def preprocess_image(image, img_size=150):
    img = tf.io.decode_image(image, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    #img /= 255.0
    #img = tf.reshape(150, 150, 3)
    img = np.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)
    #print(img.shape)

    return img


### Logic for the app flow ###
if not upload_file:
    st.warning("Please upload an image!")
    st.stop()
else:
    uploaded_image = upload_file.read() # read the uploaded image
    st.image(uploaded_image, use_column_width=True) # show the uploaded image
    pred_button = st.button("Predict")

    # make prediction, if the prediction button is click
    if pred_button == True:
        with st.spinner("Classifying..."):
            img_tensor = preprocess_image(uploaded_image)
            #print(model.summary())
            pred = model.predict(img_tensor)
            
            #"""
            #prediction = predict_json(project= PROJECT_NUMBER, 
            #                     region=REGION, 
            #                     endpoint_id=ENDPOINT_ID,
            #                     instances=img_tensor
            #                     )
            #print(prediction)
            #"""
            
            #print(pred)
            pred_class = base_classes[tf.argmax(pred[0])]
            pred_conf = tf.reduce_max(pred[0]) * 100
            
            #pred_conf = np.array(pred_conf).round(2)
            pred_conf = np.array(pred_conf)
            print(pred_conf)
            pred_conf = np.round(pred_conf, 3)
            #st.write("Predicted Class: ", pred_class )
            st.write("Predicted Class is {} and is {} confident".format(pred_class, pred_conf))
            

