import io
import re
import uuid
import base64
import os
import azure.cognitiveservices.speech as speechsdk
import streamlit as st

def clean_text_for_speech(text):
    """Clean text for speech synthesis by removing special characters"""
    cleaned_text = re.sub(r"[\*\-]", "", text)
    return cleaned_text

def text_to_speech_azure(text, azure_speech_key, azure_speech_region):
    """Convert text to speech using Azure Cognitive Services"""
    cleaned_text = clean_text_for_speech(text)
    
    try:
        # Create a speech configuration with your subscription key and region
        speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
        
        # Set the voice name (can be customized)
        speech_config.speech_synthesis_voice_name = "en-US-Ava:DragonHDLatestNeural"
        
        # Create a unique temporary file name to avoid conflicts
        temp_file = os.path.join("audio", f"temp_audio_{uuid.uuid4()}.wav")
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_file)
        
        # Create a speech synthesizer with the configured settings
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # Synthesize the text
        result = speech_synthesizer.speak_text_async(cleaned_text).get()
        
        # Check the synthesis result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Read the audio file into a BytesIO object
            with open(temp_file, 'rb') as audio_file:
                memory_stream = io.BytesIO(audio_file.read())
            
            # Reset the stream position to the start
            memory_stream.seek(0)

            # Also read the file for base64 encoding
            with open(temp_file, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Return both the stream and the filename for later cleanup, plus base64 for autoplay
            return memory_stream, temp_file, audio_base64
            
        else:
            st.error(f"Speech synthesis failed: {result.reason}")
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                st.error(f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    st.error(f"Error details: {cancellation_details.error_details}")
            
            return None, None, None
            
    except Exception as e:
        st.error(f"Azure TTS error: {e}")
        return None, None, None

def autoplay_audio(audio_base64):
    """Create an HTML audio element with autoplay"""
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html