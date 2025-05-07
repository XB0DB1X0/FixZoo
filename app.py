import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from pathlib import Path

# ----------------------------- Configuration -----------------------------
MODEL_PATH = 'my_checkpoint.weights.h5'  # ชื่อไฟล์ต้องตรงกับที่อัปขึ้น GitHub
IMG_SIZE = (224, 224)
CLASS_NAMES = [f'class_{i}' for i in range(5)]  # เปลี่ยนชื่อ class ถ้าคุณมีจริง

# ----------------------------- Model Loader -----------------------------
def build_model():
    base_model = EfficientNetB3(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None,  # สำคัญ! ต้องไม่มี weights เพราะจะโหลดจาก .h5 เอง
        pooling='avg'
    )
    x = Dense(len(CLASS_NAMES), activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=x)
    return model

@st.cache_resource(show_spinner="Loading model…")
def get_model():
    model = build_model()
    model.load_weights(MODEL_PATH)
    return model

model = get_model()

# ----------------------------- UI Config -----------------------------
st.set_page_config(page_title="Animal Classifier")
st.title("Animal Classifier Demo")
st.write("Upload an image and click Predict to classify.")

# ----------------------------- Preprocess Image -----------------------------
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# ----------------------------- Streamlit App Logic -----------------------------
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict(x, verbose=0)[0]
        top_k = preds.argsort()[-5:][::-1]

        st.subheader("Prediction (Top-5)")
        for i in top_k:
            st.write(f"- {CLASS_NAMES[i]} : {preds[i]*100:.2f}%")
