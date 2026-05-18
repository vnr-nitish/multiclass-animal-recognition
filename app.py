from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'animal_detector_model.keras'
CLASS_NAMES_PATH = BASE_DIR / 'class_names.npy'
COVER_IMAGE_PATH = BASE_DIR / 'animal_cover.png'


# Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    class_names = np.load(CLASS_NAMES_PATH, allow_pickle=False)
    return [str(name) for name in class_names.tolist()]

model = load_model()
class_names = load_class_names()

# Prediction Function
def predict_image(image_bytes):
    image = Image.open(image_bytes).convert('RGB')
    image = image.resize((224, 224))
    img = np.asarray(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    return predicted_label, confidence

# Streamlit UI
st.sidebar.title('Animal Species Classifier')
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Animal Recognition'])

# Optional splash image
if COVER_IMAGE_PATH.exists():
    st.image(str(COVER_IMAGE_PATH), use_container_width=True)

if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Animal Detection Using Deep Learning</h1>", unsafe_allow_html=True)
    st.write("Upload an image of an animal and the model will tell you what it is!")

elif app_mode == 'Animal Recognition':
    st.header("Upload an Animal Image for Recognition")
    uploaded_image = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image_bytes = uploaded_image.getvalue()

        if st.button("Show Image"):
            st.image(image_bytes, use_container_width=True)

        if st.button("Predict"):
            label, confidence = predict_image(uploaded_image)
            st.subheader("Prediction:")
            st.success(f"Model is predicting: **{label}** ({confidence * 100:.2f}% confidence)")
