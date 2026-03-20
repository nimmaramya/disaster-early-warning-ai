import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import streamlit as st


@st.cache_resource
def load_cnn_model():
    return load_model("models/final_cnn_3class_model.keras", compile=False)


def predict_cnn_risk(img_path):

    model = load_cnn_model()

    img = image.load_img(img_path, target_size=(300,300))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    normal_prob = prediction[2]
    risk_score = 1 - normal_prob

    return float(risk_score)