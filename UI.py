import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title='貓狗分類器', page_icon='🐾')
st.title('貓狗分類器')
st.write('上傳一張圖片，AI 幫你判斷是貓或狗。')

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_dog_classifier.h5')
    return model

model = load_model()
IMG_SIZE = (128,128)

uploaded_file = st.file_uploader("上傳圖片", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='上傳的圖片', use_container_width=True)
    
    if st.button("分析圖片"):
        img = image.resize(IMG_SIZE)
        img_arr = np.array(img)/255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        prob = model.predict(img_arr)[0][0]
        if prob > 0.5:
            st.success(f'判定：狗，信心度：{prob:.4f}')
        else:
            st.success(f'判定：貓，信心度：{1 - prob:.4f}')