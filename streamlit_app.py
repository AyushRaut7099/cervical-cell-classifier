import streamlit as st
import numpy as np
from PIL import Image
from skimage import filters
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input
import os, sys

MODEL_PATH = "ModelPBL.keras"   # ensure the filename matches your repo

@st.cache_resource(show_spinner=False)
def load_model_safely():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "If your model is >100MB, don't commit it to GitHub; "
            "use the download-at-startup option."
        )
    # compile=False prevents optimizer/custom-object deserialization errors
    return keras.models.load_model(MODEL_PATH, compile=False)

try:
    model = load_model_safely()
except Exception as e:
    st.error("❌ Model failed to load. See tips below.")
    st.exception(e)
    st.stop()

IMG_SIZE = (224, 224)

class_names = [
    "Dyskeratotic","Koilocytotic","Metaplastic",
    "Parabasal","Superficial-Intermediate","Cancerous","Other"
]

def is_valid_microscopic(img: Image.Image) -> bool:
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    color_std = np.std(arr, axis=2).mean()
    if color_std < 4:
        return False
    gray = np.mean(arr, axis=2).astype(np.uint8)
    ent = filters.rank.entropy(gray, np.ones((9,9))).mean()
    return ent >= 3.5

def predict_cell(img: Image.Image):
    if not is_valid_microscopic(img):
        return "Please upload a valid microscopic cervical cell image.", None
    img = img.resize(IMG_SIZE)
    x = image.img_to_array(img)[None, ...]
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds))
    return class_names[idx], round(float(np.max(preds)*100), 2)

st.title("🧬 Cervical Cell Classification")
st.write("Upload a microscope image to classify cell type (DenseNet121).")

file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded", use_column_width=True)
    label, conf = predict_cell(img)
    if conf is None:
        st.warning(label)
    else:
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{conf}%**")
