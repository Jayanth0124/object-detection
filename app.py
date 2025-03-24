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

# Set page config as the first command
st.set_page_config(page_title="Advanced AI Object Detector", layout="wide")

# Load YOLOv8 Model
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")  # Consider using "yolov8s.pt" for better performance

model = load_model()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function to get Wikipedia description
def get_wikipedia_description(obj_name):
    for _ in range(3):  # Retry up to 3 times
        try:
            summary = wikipedia.summary(obj_name, sentences=2)
            return summary
        except:
            continue
    return "No detailed description available from Wikipedia."

# Generate audio file
def generate_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = os.path.join(tempfile.gettempdir(), f"temp_{text.replace(' ', '_')}.mp3")
    tts.save(audio_file)
    return audio_file

# Draw bounding boxes with semi-transparent overlays and pose landmarks
def draw_boxes_and_pose(image, results, pose_results):
    overlay = image.copy()
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf)
        class_id = int(box.cls)
        label = f"{results[0].names[class_id]} {confidence:.2f}"
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)
    
    if pose_results and pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
        )
    
    return image

# Custom CSS for Futuristic Neumorphic Dark Mode
st.markdown("""
    <style>
    body, .stApp {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
    }
    .main {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #1a1a1a !important;
    }
    .stTabs > div, .stFileUploader, .stSlider, .stCheckbox, .stExpander {
        background: #1a1a1a;
        border-radius: 15px;
        box-shadow: 5px 5px 10px #0f0f0f, -5px -5px 10px #252525;
        padding: 15px;
        margin: 10px 0;
        color: #e0e0e0;
    }
    .stTabs button {
        background: linear-gradient(145deg, #252525, #1a1a1a);
        color: #e0e0e0;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 3px 3px 6px #0f0f0f, -3px -3px 6px #252525;
        transition: all 0.3s ease;
    }
    .stTabs button:hover {
        transform: translateY(-2px);
        box-shadow: 5px 5px 10px #0f0f0f, -5px -5px 10px #303030;
        background: linear-gradient(145deg, #303030, #252525);
    }
    .stTabs button[aria-selected="true"] {
        background: #252525;
        box-shadow: inset 3px 3px 6px #0f0f0f, inset -3px -3px 6px #303030;
    }
    .stButton>button {
        background: linear-gradient(145deg, #252525, #1a1a1a);
        color: #e0e0e0;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 3px 3px 6px #0f0f0f, -3px -3px 6px #252525;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 5px 5px 10px #0f0f0f, -5px -5px 10px #303030;
        background: linear-gradient(145deg, #303030, #252525);
    }
    .detected-object {
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .detected-object:hover {
        color: #4CAF50;
        text-shadow: 0 0 10px #4CAF50, 0 0 20px #4CAF50;
    }
    .stSlider > div > div > div {
        background: #1a1a1a;
        box-shadow: 3px 3px 6px #0f0f0f, -3px -3px 6px #252525;
    }
    .stSlider label, .stSlider div {
        color: #e0e0e0 !important;
    }
    .stCheckbox label, .stCheckbox span {
        color: #e0e0e0 !important;
    }
    .title {
        color: #e0e0e0;
        text-align: center;
        font-size: 2.5em;
        text-shadow: 2px 2px 4px #0f0f0f;
    }
    .subtitle {
        color: #b0b0b0;
        text-align: center;
        font-size: 1.2em;
    }
    [data-testid="stSidebar"] {
        background: #1a1a1a !important;
        box-shadow: 5px 5px 10px #0f0f0f;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: #e0e0e0 !important;
    }
    .stFileUploader label, .stFileUploader div {
        color: #e0e0e0 !important;
    }
    .stAlert {
        background: #252525;
        color: #e0e0e0;
        border-radius: 10px;
        box-shadow: 3px 3px 6px #0f0f0f, -3px -3px 6px #303030;
    }
    .stExpander {
        background: #1a1a1a;
        border-radius: 10px;
        box-shadow: 3px 3px 6px #0f0f0f, -3px -3px 6px #252525;
    }
    .stExpander summary, .stExpander p {
        color: #e0e0e0 !important;
    }
    [data-testid="stMarkdownContainer"] p {
        color: #e0e0e0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown('<h1 class="title">🔍 Advanced AI Object Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Capture or upload an image for object detection and pose estimation</p>', 
           unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4)
    show_descriptions = st.checkbox("Show Object Descriptions", value=True)
    show_pose = st.checkbox("Show Pose Landmarks", value=True)
    enable_voice = st.checkbox("Enable Voice Output", value=False)

# Tabs for different input modes
tab1, tab2 = st.tabs(["📸 Live Capture", "🖼️ Upload Image"])

# Live Capture Tab (Take a Photo)
with tab1:
    st.markdown('<p class="subtitle">Capture a Photo for Analysis</p>', unsafe_allow_html=True)
    st.markdown("**Note for Mobile Users:** Use the camera switch icon in your browser to toggle between front and back cameras.", unsafe_allow_html=True)
    
    if 'camera_input' not in st.session_state:
        st.session_state['camera_input'] = None

    camera_input_placeholder = st.empty()
    camera_input = camera_input_placeholder.camera_input("Take a Photo")
    if st.button("Clear Image"):
        st.session_state['camera_input'] = None
        camera_input_placeholder.empty()

    if camera_input:
        st.session_state['camera_input'] = camera_input
        with st.spinner("Processing image..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(camera_input.getvalue())
                image = cv2.imread(temp_file.name)
            if image is None:
                st.error("Failed to load the captured image. Please try again.")
                st.stop()

            # Resize image for better performance
            max_size = 640
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            results = model(image, conf=confidence_threshold)
            pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            annotated_image = draw_boxes_and_pose(image.copy(), results, pose_results if show_pose else None)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)
            with col2:
                st.image(annotated_image, caption="Detected Objects & Pose", use_container_width=True)

            if results[0].boxes:
                st.subheader("Detected Objects:")
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    obj_name = results[0].names[class_id]
                    confidence = float(box.conf)
                    if confidence >= confidence_threshold:
                        st.markdown(f'<span class="detected-object">**{obj_name.capitalize()}** (Confidence: {confidence:.2f})</span>', 
                                   unsafe_allow_html=True)
                        if enable_voice:
                            audio_file = generate_audio(f"{obj_name}")
                            if st.button(f"Play {obj_name}", key=f"play_{obj_name}_{confidence}"):
                                st.audio(audio_file, format="audio/mp3")
                            os.remove(audio_file)
                        if show_descriptions:
                            with st.expander(f"About {obj_name.capitalize()}"):
                                with st.spinner(f"Fetching description for {obj_name.capitalize()}..."):
                                    st.write(get_wikipedia_description(obj_name))
            else:
                st.warning("No objects detected above confidence threshold.")

# Upload Image Tab
with tab2:
    st.markdown('<p class="subtitle">Upload an Image for Analysis</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        with st.spinner("Analyzing image..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Failed to load the uploaded image. Please try a different image.")
                st.stop()

            # Resize image for better performance
            max_size = 640
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

            results = model(image, conf=confidence_threshold)
            pose_results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            annotated_image = draw_boxes_and_pose(image.copy(), results, pose_results if show_pose else None)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_image, caption="Detected Objects & Pose", use_container_width=True)

            if results[0].boxes:
                st.subheader("Detected Objects:")
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    obj_name = results[0].names[class_id]
                    confidence = float(box.conf)
                    if confidence >= confidence_threshold:
                        st.markdown(f'<span class="detected-object">**{obj_name.capitalize()}** (Confidence: {confidence:.2f})</span>', 
                                   unsafe_allow_html=True)
                        if enable_voice:
                            audio_file = generate_audio(f"{obj_name}")
                            if st.button(f"Play {obj_name}", key=f"play_{obj_name}_{confidence}"):
                                st.audio(audio_file, format="audio/mp3")
                            os.remove(audio_file)
                        if show_descriptions:
                            with st.expander(f"About {obj_name.capitalize()}"):
                                with st.spinner(f"Fetching description for {obj_name.capitalize()}..."):
                                    st.write(get_wikipedia_description(obj_name))
            else:
                st.warning("No objects detected above confidence threshold.")

# Footer with clickable link
st.markdown("""
    <p style='text-align:center; color: #b0b0b0;'>
    Designed by <b><a href='http://www.jayanth.xyz' target='_blank' style='color: #4CAF50; text-decoration: none;'>Donavalli Jayanth</a></b> | Powered by YOLOv8, MediaPipe & Streamlit
    </p>
""", unsafe_allow_html=True)

# Cleanup
pose.close()