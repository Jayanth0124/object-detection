import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import io
import mediapipe as mp
import wikipedia
from gtts import gTTS
import os
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set page config
st.set_page_config(page_title="Advanced AI Object Detector", layout="wide")

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")  # Use "yolov8s.pt" for better speed

model = load_model()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Wikipedia Fetch Function
def get_wikipedia_description(obj_name):
    for _ in range(3):  # Retry 3 times
        try:
            return wikipedia.summary(obj_name, sentences=2)
        except:
            continue
    return "No detailed description available."

# Audio Generation
def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_path = os.path.join(tempfile.gettempdir(), f"temp_{text.replace(' ', '_')}.mp3")
    tts.save(audio_path)
    return audio_path

# Object Detection & Pose Drawing
def draw_boxes_and_pose(image, results, pose_results):
    overlay = image.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{results[0].names[int(box.cls)]} {float(box.conf):.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if pose_results and pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return image

# Custom Futuristic Dark Theme
st.markdown("""
    <style>
    body, .stApp { background-color: #1a1a1a !important; color: #e0e0e0 !important; }
    .title { color: #e0e0e0; text-align: center; font-size: 2.5em; }
    .subtitle { color: #b0b0b0; text-align: center; font-size: 1.2em; }
    .stButton>button { background: linear-gradient(145deg, #252525, #1a1a1a); color: #e0e0e0; border-radius: 10px; }
    .stButton>button:hover { transform: translateY(-2px); background: linear-gradient(145deg, #303030, #252525); }
    .detected-object:hover { color: #4CAF50; text-shadow: 0 0 10px #4CAF50; }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<h1 class="title">🔍 Advanced AI Object Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Capture or upload an image for object detection and pose estimation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)
    show_descriptions = st.checkbox("Show Object Descriptions", value=True)
    show_pose = st.checkbox("Show Pose Landmarks", value=True)
    enable_voice = st.checkbox("Enable Voice Output", value=False)

# Tabs for input modes
tab1, tab2 = st.tabs(["📸 Live Capture", "🖼️ Upload Image"])

# Live Capture Tab
with tab1:
    st.markdown("**Note:** Switch camera mode in your browser for front/back cameras.", unsafe_allow_html=True)
    camera_input = st.camera_input("Take a Photo")
    
    if camera_input:
        with st.spinner("Processing image..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(camera_input.getvalue())
                image = cv2.imread(temp_file.name)

            if image is None:
                st.error("Failed to load the image.")
                st.stop()

            results = model(image, conf=confidence_threshold)
            pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            processed_image = draw_boxes_and_pose(image.copy(), results, pose_results if show_pose else None)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed", use_container_width=True)

# Upload Image Tab
with tab2:
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        with st.spinner("Processing image..."):
            image = Image.open(uploaded_file)
            image = np.array(image)

            results = model(image, conf=confidence_threshold)
            pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            processed_image = draw_boxes_and_pose(image.copy(), results, pose_results if show_pose else None)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                st.image(processed_image, caption="Processed", use_container_width=True)

# Display Detected Objects
if results and results[0].boxes:
    st.subheader("Detected Objects:")
    for box in results[0].boxes:
        obj_name = results[0].names[int(box.cls)]
        obj_confidence = float(box.conf)
        st.markdown(f"**🔹 {obj_name} ({obj_confidence:.2f})**", unsafe_allow_html=True)

        if show_descriptions:
            description = get_wikipedia_description(obj_name)
            st.write(description)
            if enable_voice:
                audio_path = generate_audio(description)
                st.audio(audio_path, format="audio/mp3")

st.success("🚀 Detection Completed!")
