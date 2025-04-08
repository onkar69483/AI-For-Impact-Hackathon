import cv2
import time
import numpy as np
import streamlit as st
import requests
from deepface import DeepFace
from collections import Counter

# Initialize YuNet face detector
face_detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (320, 320))
face_detector.setInputSize((640, 480))

def fetch_esp32cam_frame(esp32cam_url):
    """Fetch a frame from ESP32-CAM"""
    try:
        response = requests.get(esp32cam_url, timeout=5)
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return True, frame
    except Exception as e:
        st.error(f"❌ Error fetching image from ESP32-CAM: {e}")
        return False, None

def get_webcam_frame(cap):
    """Get a frame from laptop webcam"""
    try:
        ret, frame = cap.read()
        if ret:
            return True, frame
        else:
            st.error("❌ Failed to capture frame from webcam")
            return False, None
    except Exception as e:
        st.error(f"❌ Error capturing from webcam: {e}")
        return False, None

def test_esp32cam_connection(esp32cam_url):
    """Test the connection to ESP32-CAM"""
    success, _ = fetch_esp32cam_frame(esp32cam_url)
    return success

def test_webcam_connection():
    """Test the connection to the laptop webcam"""
    try:
        cap = cv2.VideoCapture(0)  # Default webcam (ID 0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception as e:
        st.error(f"❌ Error testing webcam: {e}")
        return False

def analyze_face_emotion(camera_source, webcam_placeholder, esp32cam_url=None):
    """Analyze emotions from face over a period of time"""
    time_frame = 10  # seconds
    start_time = time.time()
    emotion_counts = Counter()
    
    detection_status = st.empty()
    
    # Initialize webcam if needed
    webcam = None
    if camera_source == "webcam":
        webcam = cv2.VideoCapture(0)  # Default webcam (ID 0)
        if not webcam.isOpened():
            st.error("❌ Failed to open webcam")
            return "neutral"
    
    try:
        while time.time() - start_time < time_frame:
            # Get frame based on selected source
            if camera_source == "esp32cam":
                success, frame = fetch_esp32cam_frame(esp32cam_url)
            else:  # webcam
                success, frame = get_webcam_frame(webcam)
            
            if success and frame is not None:
                # Detect faces using YuNet
                frame = cv2.resize(frame, (640, 480))

                _, faces = face_detector.detect(frame)
                
                if faces is not None:
                    for face in faces:
                        x, y, w, h = map(int, face[:4])
                        face_roi = frame[y:y+h, x:x+w]
                        
                        # Convert to RGB for DeepFace
                        rgb_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        
                        try:
                            result = DeepFace.analyze(rgb_face_roi, actions=['emotion'], enforce_detection=False)
                            if isinstance(result, list) and len(result) > 0:
                                dominant_emotion = result[0]['dominant_emotion']
                                emotion_counts[dominant_emotion] += 1
                                
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                        except Exception as e:
                            st.error(f"Error analyzing emotion: {e}")
                
                webcam_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                progress_percent = int(((time.time() - start_time) / time_frame) * 100)
                detection_status.write(f"Analyzing: {progress_percent}%")
                time.sleep(0.1)
            else:
                error_source = "ESP32-CAM" if camera_source == "esp32cam" else "webcam"
                st.error(f"Failed to fetch frame from {error_source}")
                time.sleep(0.5)
    finally:
        # Make sure to release webcam if it was used
        if webcam is not None:
            webcam.release()
        
    if emotion_counts:
        most_common_emotion = emotion_counts.most_common(1)[0][0]
        return most_common_emotion
    else:
        return "neutral"