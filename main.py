import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

from detector import YOLODetector
from classifier import CLIPClassifier
from database import VectorDatabase
from utils import crop_image, extract_frames_from_video

# Initialize components
@st.cache_resource
def get_detector():
    return YOLODetector()

@st.cache_resource
def get_classifier():
    return CLIPClassifier()

@st.cache_resource
def get_database():
    return VectorDatabase()

detector = get_detector()
classifier = get_classifier()
db = get_database()

st.title("Open-World Object Detection System")

# File Upload
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4"])

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert PIL Image to OpenCV format (numpy array BGR)
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.subheader("Detecting Objects...")
        detections = detector.detect(image_np)

        if detections:
            st.subheader("Detected Objects:")
            display_image = image_np.copy()
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                conf = det['conf']

                cropped_img_pil = crop_image(image_np, det['box'])
                img_embedding = classifier.get_image_embedding(cropped_img_pil)
                
                label, distance = db.query_object(img_embedding)

                if label:
                    display_label = f"{label} ({conf:.2f})"
                else:
                    display_label = f"Unknown ({conf:.2f})"
                    st.write(f"Object {i+1} (Unknown): Please provide a label.")
                    new_label = st.text_input(f"Label for Object {i+1}", key=f"label_input_{i}")
                    if st.button(f"Save Label for Object {i+1}", key=f"save_button_{i}"):
                        if new_label:
                            db.add_object(img_embedding, new_label)
                            st.success(f"Learned: {new_label}")
                            display_label = f"{new_label} ({conf:.2f})"
                        else:
                            st.warning("Please enter a label.")

                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_image, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), caption="Objects Detected", use_column_width=True)
        else:
            st.write("No objects detected.")

    elif "video" in file_type:
        st.write("Video processing is not fully implemented yet. Displaying first frame.")
        # Save the uploaded video temporarily to process it
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        frames = extract_frames_from_video("temp_video.mp4", interval=30) # Process every 30th frame
        if frames:
            st.image(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB), caption="First Frame of Video", use_column_width=True)
            # Further processing for video frames would go here
            # For each frame, you would run detection, classification, and learning
        else:
            st.write("Could not extract frames from video.")
        
        # Clean up temporary file
        os.remove("temp_video.mp4")
