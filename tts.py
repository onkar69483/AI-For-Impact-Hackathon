import io
import uuid
import base64
import os
import azure.cognitiveservices.speech as speechsdk
import streamlit as st

# Create audio directory if it doesn't exist
if not os.path.exists("audio"):
    os.makedirs("audio")

def text_to_speech_azure(ssml, azure_speech_key, azure_speech_region):
    """Convert SSML to speech using Azure Cognitive Services"""
    try:
        # Create a speech configuration with your subscription key and region
        speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_speech_region)
        
        # Create a unique temporary file name to avoid conflicts
        temp_file = os.path.join("audio", f"temp_audio_{uuid.uuid4()}.wav")
        audio_config = speechsdk.audio.AudioOutputConfig(filename=temp_file)
        
        # Create a speech synthesizer with the configured settings
        speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        # Synthesize using SSML
        result = speech_synthesizer.speak_ssml_async(ssml).get()
        
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
    <audio autoplay controls>
        <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    return audio_html

def cleanup_temp_files(temp_file=None):
    """Clean up temporary audio files"""
    try:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
    except Exception as e:
        st.error(f"Error cleaning up temp files: {e}")

def main():
    st.set_page_config(page_title="TTS Tester", page_icon="🔊")
    
    st.title("Azure TTS Tester with SSML")
    st.write("Test Azure Text-to-Speech with SSML input")
    
    # Side panel for API configuration
    with st.sidebar:
        st.header("API Configuration")
        azure_speech_key = st.text_input("Azure Speech Key", type="password", 
                                          help="Enter your Azure Speech Services API key")
        azure_speech_region = st.text_input("Azure Speech Region", value="eastus", 
                                           help="Enter your Azure Speech Services region")
    
    # Input area for SSML
    ssml_template = """<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">
    <voice name="en-US-AvaNeural">
        <mstts:express-as style="cheerful" styledegree="1.5">
            <prosody rate="+10%" pitch="+5%" volume="+10%">
                Hello! I'm testing the <emphasis level="strong">emotional</emphasis> text-to-speech system!
            </prosody>
        </mstts:express-as>
    </voice>
</speak>"""
    ssml_input = st.text_area("Enter SSML markup", ssml_template, height=250)
    
    # Process button
    if st.button("Generate Speech"):
        if not azure_speech_key:
            st.error("Please enter your Azure Speech API key in the sidebar")
            return
        
        if not ssml_input:
            st.warning("Please enter some SSML to convert to speech")
            return
        
        with st.spinner("Generating speech..."):
            # Call the TTS service with SSML input
            audio_stream, temp_file, audio_base64 = text_to_speech_azure(
                ssml_input, azure_speech_key, azure_speech_region
            )
            
            if audio_stream:
                # Play the audio
                st.markdown(autoplay_audio(audio_base64), unsafe_allow_html=True)
                
                # Provide download button
                st.download_button(
                    label="Download Audio",
                    data=audio_stream,
                    file_name="tts_output.wav",
                    mime="audio/wav"
                )
                
                # Clean up temp file
                cleanup_temp_files(temp_file)
            else:
                st.error("Failed to generate speech. Check the logs above for details.")

if __name__ == "__main__":
    main()