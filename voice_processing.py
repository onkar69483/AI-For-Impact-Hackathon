import speech_recognition as sr
import librosa
import torch
import streamlit as st
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Load pre-trained wav2vec 2.0 model for emotion detection
EMOTION_LABELS = ["neutral", "happy", "sad", "angry"]
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")

import speech_recognition as sr
import streamlit as st

def transcribe_live_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.energy_threshold = 300  # Adjust based on noise level
        recognizer.pause_threshold = 1.2  # Waits longer before stopping
        recognizer.dynamic_energy_threshold = True  # Adjusts to background noise
        st.write("Listening... Speak continuously!")
        
        audio = recognizer.listen(source, timeout=None)  # No time limit

    try:
        transcribed_text = recognizer.recognize_google(audio)
        return transcribed_text, audio
    except sr.UnknownValueError:
        st.error("Google Web Speech API could not understand the audio.")
        return "", None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Web Speech API; {e}")
        return "", None

def analyze_emotion_from_audio(audio_file_path):
    """Analyze the emotion from an audio file using wav2vec 2.0"""
    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=16000)  # Resample to 16kHz for wav2vec 2.0
        inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)

        # Predict emotion
        with torch.no_grad():
            logits = emotion_model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        return EMOTION_LABELS[predicted_class]
    except Exception as e:
        st.error(f"Error analyzing audio emotion: {e}")
        return "neutral"