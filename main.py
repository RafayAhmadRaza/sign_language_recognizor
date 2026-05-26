import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import json
import streamlit as st
import mediapipe as mp
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import tensorflow as tf
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
VisionRunningMode = vision.RunningMode

model_path = '/home/rafayahmadraza/Machine_Learning_Projects/sign_language_recognizor/hand_landmarker.task'

tf_model_path = '/home/rafayahmadraza/Machine_Learning_Projects/sign_language_recognizor/Model/psl_fingerspell_model.keras'

model = tf.keras.models.load_model(tf_model_path)

psl_json = "/home/rafayahmadraza/Machine_Learning_Projects/sign_language_recognizor/Model/psl_labels.json"

with open(psl_json,'r') as f:
    prediction_dict = json.load(f)

@st.cache_resource
def get_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_path,
            delegate=BaseOptions.Delegate.CPU
        ),
        running_mode=VisionRunningMode.IMAGE
    )
    return HandLandmarker.create_from_options(options)

landmarker = get_landmarker()

def normalize_keypoints(kp: np.ndarray) -> np.ndarray:
    kp = kp.reshape(21, 3)
    kp = kp - kp[0]
    max_val = np.max(np.abs(kp))
    if max_val > 0:
        kp = kp / max_val
    return kp.flatten()

st.title("Pakistan Fingerspell Recognizer - Translator")


if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶ Start Camera"):
        st.session_state.running = True
with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.running = False

frame_placeholder = st.empty()
sign_placeholder = st.empty()

if st.session_state.running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    frame_count = 0

    while st.session_state.running:
        ret, img = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        frame_count += 1

        if frame_count % 3 == 0:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
                normalized = normalize_keypoints(kp)

                predicted_value = model.predict(normalized.reshape(1,-1), verbose=0)
                predicted_index = str(np.argmax(predicted_value[0]))

                h, w, _ = img.shape
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 4, (0, 255, 0), -1)

                sign_placeholder.text_area(
                    "Predicted Sign",
                    value=str(prediction_dict[predicted_index]),
                    disabled=True,
                    key=f"sign_{frame_count}" 
                )

        frame_placeholder.image(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

        time.sleep(0.033)  # ~30fps

    cap.release()