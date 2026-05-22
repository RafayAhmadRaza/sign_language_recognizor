import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from streamlit_webrtc import webrtc_streamer

model_path = '/hand_landmarker.task'

st.title("Pakistan Fingerspell Recognizer - Translator")

webrtc_streamer(key="streamer",sendback_audio=False)