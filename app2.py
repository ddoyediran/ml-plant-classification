import streamlit as st
#import SessionState
import tensorflow as tf
import requests



st.title("Welcome to Plant Pathology Kaggle challenge")
st.header("Identify plant leaves diseases")
st.header("A Deep learning app to help identify if an apple leaf is healthy, has multiple diseases, rust or scab")


# File uploader allows user to add their own image
upload_file = st.file_uploader(label="Please upload image of the apple leaf", 
                               type=["png", "jpg", "jpeg"])

### Function to load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_class.keras")
    return model

# Show spinner on loading
with st.spinner("Loading model..."):
    model = load_model()

# Classes and Preprocess the image
classes = ['healthy',
 'multiple_diseases',
 'rust',
 'scab'
 ]

def load_image(image, img_size=150):
    img = tf.io.decode_image(image, channels=3)
    img = tf.cast(img, tf.float32)
    img /= 255.0
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.expand_dims(img, axis=0)
    print(img)
    return img


### Logic for the app flow ###
if not upload_file:
    st.warning("Please upload an image!")
    st.stop()
else:
    uploaded_image = upload_file.read() # read the uploaded image
    st.image(uploaded_image, use_column_width=True) # show the uploaded image
    pred_button = st.button("Predict")
    # print(pred_button) #
    if pred_button == True:
        with st.spinner("Classifying..."):
            img_tensor = load_image(uploaded_image)
            pred = model.predict(img_tensor)
            print(pred)
            pred_class = classes[tf.argmax(pred[0])]
            print(tf.argmax(pred[0]))
            st.write("Predicted Class: ", pred_class)
            #st.image(uploaded_image, use_column_width=True)

