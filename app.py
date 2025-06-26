import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input


# Constants
IMG_SIZE = 224
CLASS_NAMES = ['hazardous', 'organic', 'recyclable']

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('waste_classifier_model.h5')

model = load_model()

# Preprocess image
def preprocess_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))               # Resize to 224x224
    img = np.array(img).astype(np.float32)                 # Convert to float32
    img = preprocess_input(img)                            # EfficientNetB0 preprocessing
    return np.expand_dims(img, axis=0)                     # Add batch dimension


# Predict function
def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]

    st.subheader("🔍 Prediction Probabilities:")
    for i, prob in enumerate(prediction):
        st.write(f"- {CLASS_NAMES[i].capitalize()}: **{prob * 100:.2f}%**")

    idx = np.argmax(prediction)
    return CLASS_NAMES[idx], float(np.max(prediction))


# Streamlit UI
st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️")
st.title("♻️ Smart Waste Classification System")
st.write("Upload an image of **waste**, and the AI will classify it as:")
st.markdown("""
- ✅ **Hazardous**  
- 🌿 **Organic**  
- ♻️ **Recyclable**
""")

uploaded_file = st.file_uploader("📤 Upload a waste image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner('🧠 Analyzing...'):
        label, confidence = predict(image)

    st.success(f"🗂️ **Prediction:** `{label.upper()}` with **{confidence*100:.2f}%** confidence")

