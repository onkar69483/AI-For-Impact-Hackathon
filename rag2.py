# Necessary libraries
import os
import streamlit as st
import io
import time
import uuid
from dotenv import load_dotenv
import tensorflow as tf
import cv2
import pickle

# LangChain libraries for RAG
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# Custom modules
from face_emotion import test_esp32cam_connection, fetch_esp32cam_frame, analyze_face_emotion, test_webcam_connection
from voice_processing import transcribe_live_audio, analyze_emotion_from_audio
from text_analysis import analyze_sentiment
from response_generation import generate_empathetic_response
from tts_services import text_to_speech_azure, autoplay_audio
from utils import reset_session


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file. Please add it and restart the app.")
    st.stop()

# Azure Speech Service credentials
azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
if not azure_speech_key or not azure_speech_region:
    st.error("❌ AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not found in .env file. Please add them and restart the app.")
    st.stop()

ESP32CAM_URL = "http://192.168.127.99/640x480.jpg"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("❌ HF_TOKEN not found in .env file. Please add it and restart the app.")
    st.stop()
os.environ['HF_TOKEN'] = hf_token

audio_dir = "audio"
pdf_dir = "pdf_uploads"
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
if not os.path.exists(pdf_dir):
    os.makedirs(pdf_dir)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        st.error(f"GPU error: {e}")


# Initialize session state
def init_session_state():
    # Emotion detection states
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "face_emotion" not in st.session_state:
        st.session_state.face_emotion = None
    if "voice_emotion" not in st.session_state:
        st.session_state.voice_emotion = None
    if "final_emotion" not in st.session_state:
        st.session_state.final_emotion = None
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "start_face_detection" not in st.session_state:
        st.session_state.start_face_detection = False
    if "start_voice_recording" not in st.session_state:
        st.session_state.start_voice_recording = False
    if "temp_files" not in st.session_state:
        st.session_state.temp_files = []
    if "esp32cam_connected" not in st.session_state:
        st.session_state.esp32cam_connected = False
    if "webcam_connected" not in st.session_state:
        st.session_state.webcam_connected = False
    if "audio_data" not in st.session_state:
        st.session_state.audio_data = None
    if "esp32cam_url" not in st.session_state:
        st.session_state.esp32cam_url = ESP32CAM_URL
    if "camera_source" not in st.session_state:
        st.session_state.camera_source = "esp32cam"
        
    # RAG states
    if 'rag_store' not in st.session_state:
        st.session_state.rag_store = {}
    if 'session_id' not in st.session_state:
        st.session_state.session_id = "default_session"
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'conversational_rag_chain' not in st.session_state:
        st.session_state.conversational_rag_chain = None
    if 'rag_response' not in st.session_state:
        st.session_state.rag_response = None
    if 'used_rag' not in st.session_state:
        st.session_state.used_rag = False

# Initialize LLM and embeddings
def init_llm_and_embeddings():
    # Initialize Groq LLM
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return llm, embeddings
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        st.stop()

def process_pdfs(embeddings):
    existing_pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    if existing_pdfs:
        with st.spinner("Processing PDFs..."):
            documents = []

            # Load PDFs from directory
            for pdf_path in existing_pdfs:
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Error loading {pdf_path}: {e}")

            # Split and create embeddings for documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)

            # Use FAISS for vector storage
            vectorstore = FAISS.from_documents(splits, embedding=embeddings)
            faiss_index_path = "faiss_index.pkl"
            with open(faiss_index_path, "wb") as f:
                pickle.dump(vectorstore, f)

            return vectorstore
    else:
        st.info("No PDFs found. RAG functionality will be disabled.")
        return None
    

def setup_rag_chain(vectorstore, llm):
    if vectorstore is None:
        return None
    
    retriever = vectorstore.as_retriever()
    
    # Contextualization prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Reformulate the user query into a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Question-answering prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Use retrieved context to answer concisely:\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.rag_store:
        st.session_state.rag_store[session] = ChatMessageHistory()
    return st.session_state.rag_store[session]


def create_conversational_rag_chain(rag_chain):
    if rag_chain is None:
        return None
        
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

def query_rag_system(user_input, session_id):
    if st.session_state.conversational_rag_chain is None:
        return None, False
    
    try:
        response = st.session_state.conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        
        # Check if response is relevant
        answer = response.get('answer', '')
        
        # Simple heuristic to check if the answer is substantive
        # You might want to improve this with a relevance score







        # CHECK THIS CONDITION







        if len(answer) > 50 and "I don't know" not in answer.lower() and "I don't have" not in answer.lower():
            return answer, True
        return None, False
    except Exception as e:
        st.error(f"Error querying RAG system: {e}")
        return None, False
    

def main():
    st.title("Hope")
    init_session_state()
    llm, embeddings = init_llm_and_embeddings()

    # if st.session_state.vectorstore is None:
    st.session_state.vectorstore = process_pdfs(embeddings=embeddings) # inside if?

    if st.session_state.rag_chain is None and st.session_state.vectorstore is not None:
        st.session_state.rag_chain = setup_rag_chain(st.session_state.vectorstore, llm)
        st.session_state.conversational_rag_chain = create_conversational_rag_chain(st.session_state.rag_chain)

    st.sidebar.header("Camera Configuration")

    camera_source = st.sidebar.radio(
        "Select Camera Source", 
        ["ESP32-CAM", "Laptop Webcam"], 
        index=0 if st.session_state.camera_source == "esp32cam" else 1
    )

    st.session_state.camera_source = "esp32cam" if camera_source == "ESP32-CAM" else "webcam"

    if st.session_state.camera_source == "esp32cam":
        esp32_cam_url = st.sidebar.text_input("ESP32-CAM URL", st.session_state.esp32cam_url)
        
        # Update the ESP32-CAM URL if changed
        if esp32_cam_url != st.session_state.esp32cam_url:
            st.session_state.esp32cam_url = esp32_cam_url
        
        # Test ESP32-CAM connection
        if st.sidebar.button("Test ESP32-CAM Connection"):
            if test_esp32cam_connection(st.session_state.esp32cam_url):
                st.sidebar.success("✅ Successfully connected to ESP32-CAM!")
                st.session_state.esp32cam_connected = True
            else:
                st.sidebar.error("❌ Failed to connect to ESP32-CAM. Please check the URL and connection.")
                st.session_state.esp32cam_connected = False

    else:
    # Test webcam connection
        if st.sidebar.button("Test Webcam Connection"):
            if test_webcam_connection():
                st.sidebar.success("✅ Successfully connected to webcam!")
                st.session_state.webcam_connected = True
            else:
                st.sidebar.error("❌ Failed to connect to webcam. Please check if it's available and not in use.")
                st.session_state.webcam_connected = False

    st.sidebar.header("Document Knowledge Base")
    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id

    if st.sidebar.button("New RAG Session"):
        new_session_id = f"session_{len(st.session_state.rag_store) + 1}"
        st.session_state.session_id = new_session_id
        st.session_state.rag_store[new_session_id] = ChatMessageHistory()
        st.sidebar.success(f"New session created: {new_session_id}")


    if st.button("Reset Pipeline", key="reset_pipeline_button"):
        reset_session()
        st.rerun()

    # Progress bar to show the current step
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(int((st.session_state.step - 1) / 3 * 100))


    # Step 1: Face Emotion Detection
    if st.session_state.step == 1:
        progress_bar.progress(33)
        st.subheader("Step 1: Face Emotion Detection")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            webcam_placeholder = st.empty()
        
        with col2:
            if st.button("Start Face Detection", key="start_face_detection_button"):
                if st.session_state.camera_source == "esp32cam":
                    if not test_esp32cam_connection(st.session_state.esp32cam_url):
                        st.error("❌ Cannot connect to ESP32-CAM. Please check the connection and URL.")
                    else:
                        st.session_state.start_face_detection = True
                        st.session_state.esp32cam_connected = True
                else:  # webcam
                    if not test_webcam_connection():
                        st.error("❌ Cannot connect to webcam. Please check if it's available and not in use.")
                    else:
                        st.session_state.start_face_detection = True
                        st.session_state.webcam_connected = True
        
        if st.session_state.start_face_detection:
            if (st.session_state.camera_source == "esp32cam" and st.session_state.esp32cam_connected) or \
               (st.session_state.camera_source == "webcam" and st.session_state.webcam_connected):
                
                emotion = analyze_face_emotion(
                    st.session_state.camera_source,
                    webcam_placeholder,
                    esp32cam_url=st.session_state.esp32cam_url
                )
                
                if emotion:
                    st.session_state.face_emotion = emotion
                    st.success(f"Detected emotion from face: {emotion}")
                else:
                    st.session_state.face_emotion = "neutral"
                    st.warning("No emotions detected, using neutral as default.")
                
                st.session_state.step = 2
                st.session_state.start_face_detection = False
                st.rerun()

    # Step 2: Voice Emotion Detection
    elif st.session_state.step == 2:
        progress_bar.progress(66)
        st.subheader("Step 2: Voice Emotion Detection")
        
        st.write(f"Face emotion: {st.session_state.face_emotion}")
        
        if st.button("Start Voice Recording", key="start_voice_recording_button"):
            st.session_state.start_voice_recording = True

        if st.session_state.start_voice_recording:
            user_input, audio = transcribe_live_audio()
            
            if user_input:
                st.write("Transcribed Text:", user_input)
                st.session_state.user_input = user_input
                
                if audio:
                    # Generate unique audio filename
                    audio_temp_file = os.path.join("audio", f"voice_temp_{uuid.uuid4()}.wav")
                    with open(audio_temp_file, "wb") as f:
                        f.write(audio.get_wav_data())
                    
                    # Add to list of files to clean up later
                    st.session_state.temp_files.append(audio_temp_file)
                    
                    voice_emotion = analyze_emotion_from_audio(audio_temp_file)
                    st.session_state.voice_emotion = voice_emotion
                    st.success(f"Detected emotion from voice: {voice_emotion}")
                    
                    st.session_state.step = 3
                    st.session_state.start_voice_recording = False
                    st.rerun()
        
    elif st.session_state.step == 3:
        progress_bar.progress(100)
        st.subheader("Step 3: Final Result and Response")
        
        # Display detected emotions
        if st.session_state.face_emotion:
            st.write(f"Face emotion: {st.session_state.face_emotion}")
        
        if st.session_state.voice_emotion:
            st.write(f"Voice emotion: {st.session_state.voice_emotion}")
        
        # Analyze sentiment from transcribed text if we have user input
        text_sentiment = None
        if st.session_state.user_input:
            text_sentiment = analyze_sentiment(st.session_state.user_input)
            st.write(f"Text sentiment: {text_sentiment}")


        if not st.session_state.processing_complete:
            st.session_state.conversation.append({"role": "user", "content": st.session_state.user_input})
            
            # First try to get an answer from the RAG system
            rag_answer = None
            used_rag = False
            
            if st.session_state.vectorstore is not None:
                with st.spinner("Checking document knowledge base..."):
                    rag_answer, used_rag = query_rag_system(st.session_state.user_input, st.session_state.session_id)
                    st.session_state.rag_response = rag_answer
                    st.session_state.used_rag = used_rag
            
            # Generate response based on whether we got a RAG answer or not
            if used_rag and rag_answer:
                # If we have a relevant answer from RAG, we still generate an empathetic response
                # but include the document knowledge in context
                response = generate_empathetic_response(
                    st.session_state.user_input,
                    st.session_state.conversation,
                    face_emotion=st.session_state.face_emotion,
                    voice_emotion=st.session_state.voice_emotion,
                    text_sentiment=text_sentiment,
                    rag_information=rag_answer  # Pass the RAG answer as additional context
                )
                st.info("Response includes information from your documents")
            else:
                # Regular empathetic response without document knowledge
                response = generate_empathetic_response(
                    st.session_state.user_input,
                    st.session_state.conversation,
                    face_emotion=st.session_state.face_emotion,
                    voice_emotion=st.session_state.voice_emotion,
                    text_sentiment=text_sentiment
                )

            st.session_state.conversation.append({"role": "assistant", "content": response})
            st.session_state.processing_complete = True
            
            st.write("*AI Response:*", response)

            audio_file, temp_filename, audio_base64 = text_to_speech_azure(response, azure_speech_key, azure_speech_region)
            if audio_file and audio_base64:
                # Store the base64 audio data in session state
                st.session_state.audio_data = audio_base64
                
                # Display standard audio widget (as fallback)
                st.audio(audio_file, format="audio/wav")
                
                # Display autoplay HTML component
                autoplay_container = st.empty()
                autoplay_container.markdown(autoplay_audio(audio_base64), unsafe_allow_html=True)
                
                # Store the filename to clean up later
                if temp_filename:
                    st.session_state.temp_files.append(temp_filename)
            
            if st.button("Start New Interaction", key="start_new_interaction_button"):
                reset_session()
                st.rerun()
    
    st.sidebar.header("Live Camera Feed")
    feed_label = "ESP32-CAM" if st.session_state.camera_source == "esp32cam" else "Webcam"
    if st.sidebar.button(f"Show Live {feed_label} Feed"):
        live_feed = st.sidebar.empty()
        stop_feed = st.sidebar.button("Stop Live Feed")
        
        if st.session_state.camera_source == "esp32cam":
            # ESP32-CAM feed
            while not stop_feed:
                success, frame = fetch_esp32cam_frame(st.session_state.esp32cam_url)
                if success and frame is not None:
                    live_feed.image(frame, channels="BGR", width=320)
                    time.sleep(0.1)
                else:
                    live_feed.error("Failed to fetch frame")
                    time.sleep(1)
                    break
        else:
            # Webcam feed
            cap = cv2.VideoCapture(0)  # Default webcam
            try:
                while not stop_feed and cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        live_feed.image(frame, channels="BGR", width=320)
                        time.sleep(0.1)
                    else:
                        live_feed.error("Failed to capture frame")
                        time.sleep(1)
                        break
            finally:
                cap.release()

    # Conversation history
    st.subheader("Conversation History")
    history_container = st.container()
    with history_container:
        for message in st.session_state.conversation:
            if message["role"] == "user":
                st.write(f"*You:* {message['content']}")
            else:
                st.write(f"*AI:* {message['content']}")

    # Export chat history 
    if st.button("Export Chat History"):
        chat_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation])
        st.download_button(
            label="Download Chat History",
            data=chat_history_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

