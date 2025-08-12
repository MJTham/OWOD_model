import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import imageio

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

# Pre-populate the database with YOLO class names
with st.spinner("Loading model classes into database..."):
    class_names = detector.get_class_names()
    for class_id, name in class_names.items():
        # Check if the class name is already in the database
        if not db.query_object_by_label(name):
            text_embedding = classifier.get_text_embedding(name)
            db.add_object(text_embedding, name)

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

        with st.spinner("Detecting objects in image..."):
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
        st.subheader("Processing Video...")
        temp_video_path = "temp_video.mp4"
        try:
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"Error saving video file: {e}")
            st.stop()

        with st.spinner("Extracting frames and processing video... This may take a while."):
            frames, original_fps = extract_frames_from_video(temp_video_path)
            
            if frames:
                st.write(f"Extracted {len(frames)} frames. Processing each frame...")
                processed_frames = []
                for i, frame in enumerate(frames):
                    # Convert frame to RGB for PIL and then back to BGR for OpenCV if needed
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    detections = detector.detect(frame)
                    
                    display_frame = frame.copy()
                    if detections:
                        for det in detections:
                            x1, y1, x2, y2 = det['box']
                            conf = det['conf']
                            
                            cropped_img_pil = crop_image(frame, det['box'])
                            img_embedding = classifier.get_image_embedding(cropped_img_pil)
                            
                            label, distance = db.query_object(img_embedding)
                            print(f"Query result: label={label}, distance={distance}")
                            
                            if label:
                                display_label = f"{label} ({conf:.2f})"
                            else:
                                display_label = f"Unknown ({conf:.2f})"

                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    processed_frames.append(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                
                if processed_frames:
                    # Create a video from processed frames
                    output_video_path = "output.mp4"
                    imageio.mimwrite(output_video_path, processed_frames, fps=original_fps, format='mp4')

                    # Read the video file into memory
                    with open(output_video_path, "rb") as f:
                        video_bytes = f.read()

                    st.subheader("Processed Video:")
                    st.video(video_bytes)

                else:
                    st.write("No frames were processed to create a video.")
            else:
                st.write("Could not extract frames from video.")
        
        # Clean up temporary files
        os.remove(temp_video_path)

        
