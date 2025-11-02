import streamlit as st
import numpy as np
from PIL import Image
from skimage import filters
from tensorflow.keras.preprocessing import image
from tensorflow import keras
from tensorflow.keras.applications.densenet import preprocess_input

# Load model once
@st.cache_resource
def load_model():
    return keras.models.load_model("ModelPBL.keras")

model = load_model()
IMG_SIZE = (224, 224)

class_names = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
    "Cancerous",
    "Other"
]

def is_valid_microscopic(img):
    img_arr = np.array(img)

    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return False

    color_std = np.std(img_arr, axis=2).mean()
    if color_std < 4:
        return False

    gray = np.mean(img_arr, axis=2).astype(np.uint8)
    entropy = filters.rank.entropy(gray, np.ones((9, 9))).mean()
    if entropy < 3.5:
        return False

    return True

def predict_cell(img):
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    idx = np.argmax(preds)
    return class_names[idx], round(np.max(preds) * 100, 2)

# App UI
st.title("🧬 Cervical Cell Classification App")
st.write("Upload a microscope image to predict the type of cervical cell using an AI model trained on the Herlev dataset.")

file = st.file_uploader("Upload Microscopic Image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if not is_valid_microscopic(img):
        st.error("❌ Please upload a valid microscopic cervical cell image.")
    else:
        label, conf = predict_cell(img)
        st.success(f"✅ Predicted: **{label}**")
        st.info(f"Confidence: **{conf}%**")
