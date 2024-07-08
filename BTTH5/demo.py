import streamlit as st
import numpy as np
import cv2
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import joblib

# Load model
featureExtractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def classify_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (150,150))
    image = np.array(image).flatten() / 255.0

    model = joblib.load('svm_model.pkl')
    
    # Predict
    pred = model.predict(image.reshape(1, -1))
    if pred[0] == 0:
        return "Normal"
    else:
        return "Pneumonia"

def main():
    st.title('Demo Chest X-ray Classification')
    
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Đọc file ảnh từ file_uploader
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Hiển thị ảnh gốc
        st.subheader("Ảnh gốc")
        st.image(image, channels="BGR")

        # Xử lý ảnh
        out = classify_image(image)

        # Hiển thị ảnh sau khi xử lý
        st.subheader("Kết quả phân loại")
        st.write(out)

if __name__ == "__main__":
    main()