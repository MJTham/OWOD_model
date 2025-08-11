import cv2
import numpy as np
from PIL import Image

def crop_image(image: np.ndarray, box: list):
    """
    Crops an image based on bounding box coordinates.
    Args:
        image (np.ndarray): The input image in OpenCV format (BGR).
        box (list): Bounding box coordinates [x1, y1, x2, y2].
    Returns:
        PIL.Image.Image: The cropped image as a PIL Image.
    """
    x1, y1, x2, y2 = map(int, box)
    cropped_np = image[y1:y2, x1:x2]
    return Image.fromarray(cv2.cvtColor(cropped_np, cv2.COLOR_BGR2RGB))

def extract_frames_from_video(video_path: str, interval: int = 1):
    """
    Extracts frames from a video file.
    Args:
        video_path (str): Path to the video file.
        interval (int): Interval at which to extract frames (e.g., 1 for every frame, 30 for every 30th frame).
    Returns:
        list: A list of numpy arrays, each representing a frame.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

if __name__ == '__main__':
    # Example usage for crop_image:
    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_image[10:50, 20:60] = [0, 0, 255] # Draw a red rectangle
    box = [20, 10, 60, 50]
    cropped_img = crop_image(dummy_image, box)
    print(f"Cropped image size: {cropped_img.size}")
    # cropped_img.save("cropped_example.png") # Uncomment to save and view

    # Example usage for extract_frames_from_video (requires a dummy video file)
    # You would need a video file for this to work.
    # dummy_video_path = "dummy_video.mp4"
    # frames = extract_frames_from_video(dummy_video_path, interval=10)
    # print(f"Extracted {len(frames)} frames.")
