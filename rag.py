import streamlit as st
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
import os
import pickle
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
os.environ['HF_TOKEN'] = hf_token

# Initialize embeddings model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Streamlit UI setup
st.title("📚 Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDFs and chat with their content. Powered by Groq and LangChain.")

# Validate API key
if not api_key:
    st.error("Groq API Key is missing. Please set it in your environment.")
    st.stop()

# Initialize Groq LLM
try:
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
except Exception as e:
    st.error(f"Error initializing Groq model: {e}")
    st.stop()

# Session management
if 'store' not in st.session_state:
    st.session_state.store = {}

session_id = st.text_input("Session ID", value="default_session")
if st.button("New Session"):
    session_id = f"session_{len(st.session_state.store) + 1}"
    st.session_state.store[session_id] = ChatMessageHistory()
    st.success(f"New session created: {session_id}")

# Directory setup
pdf_dir = "pdf_uploads"
os.makedirs(pdf_dir, exist_ok=True)

# Check for existing PDFs
existing_pdfs = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Process uploaded PDFs
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

        retriever = vectorstore.as_retriever()
    
    st.success("PDFs processed and embeddings generated!")

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

    # Function to get session history
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    # Define conversational RAG chain
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # User question input
    user_input = st.text_input("Your question:")
    if user_input:
        with st.spinner("Generating response..."):
            session_history = get_session_history(session_id)
            try:
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}}
                )
                st.write("### Assistant:")
                st.write(response['answer'])
            except Exception as e:
                st.error(f"Error generating response: {e}")

        st.write("### Chat History:")
        for message in session_history.messages:
            st.write(f"- {message}")

    # Export chat history
    if st.button("Export Chat History"):
        chat_history_text = "\n".join([str(msg) for msg in session_history.messages])
        st.download_button(
            label="Download Chat History",
            data=chat_history_text,
            file_name="chat_history.txt",
            mime="text/plain"
        )

else:
    st.warning("No PDFs found. Please upload one to continue.")

# Run the application
if __name__ == "__main__":
    st.write("Ready for conversation with uploaded PDFs!")
