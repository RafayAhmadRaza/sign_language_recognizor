import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from streamlit_webrtc import webrtc_streamer

mp_hands =  mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,max_num_hands=1)

st.title("Pakistan Fingerspell Recognizer - Translator")

webrtc_streamer(key="streamer",sendback_audio=False)