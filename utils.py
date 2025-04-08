import os
import streamlit as st

def reset_session():
    """Reset all session state variables"""
    # Store camera configuration settings to preserve them
    camera_source = st.session_state.get("camera_source", "esp32cam")
    esp32cam_url = st.session_state.get("esp32cam_url", "http://192.168.127.99/640x480.jpg")
    esp32cam_connected = st.session_state.get("esp32cam_connected", False)
    webcam_connected = st.session_state.get("webcam_connected", False)
    
    # Reset emotion detection variables
    st.session_state.face_emotion = None
    st.session_state.voice_emotion = None
    st.session_state.final_emotion = None
    st.session_state.user_input = ""
    st.session_state.processing_complete = False
    st.session_state.step = 1
    st.session_state.start_face_detection = False
    st.session_state.start_voice_recording = False
    st.session_state.audio_data = None
    
    # Clean up temporary files
    if "temp_files" in st.session_state:
        for temp_file in st.session_state.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                pass  # Ignore errors if file is still in use
        st.session_state.temp_files = []
    
    # Restore camera settings
    st.session_state.camera_source = camera_source
    st.session_state.esp32cam_url = esp32cam_url
    st.session_state.esp32cam_connected = esp32cam_connected
    st.session_state.webcam_connected = webcam_connected

    st.session_state.rag_response = None
    st.session_state.used_rag = False

