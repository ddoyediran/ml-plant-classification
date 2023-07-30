import os
import streamlit as st
#import SessionState
#import tensorflow as tf
#import requests
#from utils import load_and_prep_image, classes_and_models, update_logger, predict_json
from utils import load_and_prep_image, update_logger

# Setup environment credentials (you'll need to change these)
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "daniels-dl-playground-4edbcb2e6e37.json" # change for your GCP key
#PROJECT = "daniels-dl-playground" # change for your GCP project
#REGION = "us-central1" # change for your GCP region (where your model is hosted)

st.title("Welcome to Plant Pathology Kaggle challenge")
st.header("Identify plant leaves diseases")
st.header("A Deep learning app to help identify if an apple leaf is healthy, has multiple diseases, rust or scab")

@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image, model, class_names):
    """
    Takes an image and uses model (a trained TensorFlow model) to make a
    prediction.

    Returns:
     image (preprocessed)
     pred_class (prediction class from class_names)
     pred_conf (model confidence)
    """

    image = load_and_prep_image(image)
    #image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    #pred = predict_json(project=PROJECT,
    #                    region=REGION,
    #                    model=model
    #                    )
    #pred_class = class_names[tf.argmax(pred[0])]
    #pred_conf = tf.reduce_max(preds[0])

    #return image, pred_class, pred_conf

# File uploader allows user to add their own image
upload_file = st.file_uploader(label="Please upload image of the apple leaf", 
                               type=["png", "jpg", "jpeg"])


### Logic for the app flow ###
if not upload_file:
    st.warning("Please upload an image!")
    st.stop()
else:
    st.image(upload_file, use_column_width=True)
    pred_button = st.button("Predict")

# If user click the predict button
#if pred_button:
#    session_state.pred_button = True