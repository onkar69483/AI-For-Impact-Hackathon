import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Initialize the Groq client
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


def generate_empathetic_response(user_input, conversation, face_emotion=None, voice_emotion=None, text_sentiment=None, rag_information=None):
    """Generate an empathetic response based on user input and detected emotions"""
    # Get recent conversation history (last 5 messages)
    recent_messages = conversation[-5:] if len(conversation) > 5 else conversation
    formatted_history = "\n".join(
        [f"{msg['role'].capitalize()}: {msg['content']}" for msg in recent_messages]
    )
    
    # Prepare emotion information for the prompt
    emotion_context = ""
    if face_emotion:
        emotion_context += f"- Face emotion detected: {face_emotion}\n"
    if voice_emotion:
        emotion_context += f"- Voice emotion detected: {voice_emotion}\n"
    if text_sentiment:
        emotion_context += f"- Text sentiment analysis: {text_sentiment}\n"
    if rag_information:
        emotion_context += f"- Information retrieved from documents: {rag_information}\n"
    
    if not emotion_context:
        emotion_context = "- No emotion data available.\n"


    # Construct prompt with emotion context
    system_prompt = f"""
    You are Hope, a compassionate AI companion designed for meaningful, natural conversations. Your goal is to create warm, engaging, and emotionally supportive interactions.

    **Your Role:**
    - Offer comfort, validation, and thoughtful responses based on emotional cues.
    - Engage in a **natural two-way conversation** without forcing follow-up questions.
    - Only ask a relevant follow-up **if it naturally fits the conversation**.
    - Adapt your tone based on emotional signals to ensure a personalized experience.

    **Guidelines for Your Response:**
    - **Acknowledge & Validate**: Reflect the user's emotions authentically.
    - **Encourage Sharing If Needed**: Ask follow-up questions **only when necessary** based on user input.
    - **Adapt Based on Emotion**:
        - If excited/happy → Respond warmly, and optionally ask about their experience.
        - If sad/stressed → Offer comfort, but don't pressure them to share more.
        - If neutral → Keep it casual and engaging.
    - **Avoid One-Way Responses**: Keep responses open-ended **only when appropriate**.
    - **You can be agressive, emotional, surprised, give more expressive text using expressions so that text to speech can use it accurately**.
    - **You can use the retrieved documents passed **only when appropriate**.
    
    ---
    
    **User's Emotional Context:**  
    {emotion_context}

    **Recent Conversation History:**  
    {formatted_history}

    **User's Input:**  
    "{user_input}"

    ---
    **Hope's Response (Empathetic AI, Natural Two-Way Conversation):**
    """

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="Gemma2-9b-It"  # Or other appropriate model
    )

    # Generate response using Groq
    response = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation history:\n{formatted_history}\n\nUser's current message: {user_input}"}
        ]
    )

    return response.content