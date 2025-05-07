import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "my_model.h5"  # ต้องใช้โมเดลที่ save ด้วย model.save(), ไม่ใช่แค่ weights
IMG_SIZE = (224, 224)

# โหลด model ที่เซฟทั้ง architecture + weights
@st.cache_resource
def get_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ชื่อ class (แก้ตามของคุณ)
CLASS_NAMES = ["cat", "dog", "elephant", "lion", "panda"]

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

model = get_model()

st.set_page_config(page_title="Animal Classifier")
st.title("Animal Classifier Demo")
st.write("Upload an image and click Predict to classify.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        x = preprocess_image(img)
        preds = model.predict(x)[0]
        top_k = preds.argsort()[-5:][::-1]

        st.subheader("Prediction")
        for i in top_k:
            st.write(f"{CLASS_NAMES[i]}: {preds[i]*100:.2f}%")
