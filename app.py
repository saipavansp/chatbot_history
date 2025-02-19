import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI  # Fallback option if needed
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
import time
from typing import Optional


# Add CSS for chat styling
def add_chat_styling():
    st.markdown("""
        <style>        
        .human-msg {
            background-color: #075E54;
            color: white;
            float: right;
            border-radius: 15px 15px 0 15px;
            padding: 10px 15px;
            margin: 5px;
            display: inline-block;
        }
        .assistant-msg {
            background-color: #128C7E;
            color: white;
            float: left;
            border-radius: 15px 15px 15px 0;
            padding: 10px 15px;
            margin: 5px;
            display: inline-block;
        }
        # .chat-background {
        #     background-color: #ECE5DD;
        #     padding: 20px;
        #     border-radius: 10px;
        #     margin: 10px 0;
        #     min-height: 400px;
        #     overflow-y: auto;
        # }
        </style>
    """, unsafe_allow_html=True)


# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# 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 BFSI key
# Use the provided Groq API key (hardcoded)
GROQ_API_KEY = "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"


def initialize_llm(api_key: str, retries: int = 3) -> Optional[object]:
    """Initialize LLM with retry logic and fallback options."""
    for attempt in range(retries):
        try:
            return ChatGroq(groq_api_key=api_key, model="llama-3.3-70b-versatile")
        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                st.warning(f"Failed to initialize Groq after {retries} attempts. Error: {str(e)}")
                openai_key = st.text_input("Groq unavailable. Enter OpenAI API key for fallback:", type="password")
                if openai_key:
                    try:
                        return ChatOpenAI(api_key=openai_key)
                    except Exception as openai_error:
                        st.error(f"Failed to initialize OpenAI fallback. Error: {str(openai_error)}")
                        return None
            time.sleep(1)
    return None


# Initialize chat display
def display_chat_messages(session_history):
    for message in session_history.messages:
        if message.type == 'human':
            st.markdown(f'<div class="chat-container"><div class="human-msg">{message.content}</div></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-container"><div class="assistant-msg">{message.content}</div></div>',
                        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Set up Streamlit interface
st.title("Conversational RAG With PDF Uploads")
st.write("Upload PDFs and ask your question. The assistant's answer will be shown below.")

# Add chat styling
add_chat_styling()

# Initialize the language model
llm = initialize_llm(GROQ_API_KEY)

if llm is None:
    st.error("Failed to initialize any language model. Please try again later.")
else:
    # Let the user specify a session ID
    session_id = st.text_input("Session ID", value="default_session")

    # Initialize chat history store if not already present
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Upload PDF files
    uploaded_files = st.file_uploader("Choose PDF file(s)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        try:
            import tempfile

            documents = []

            # Create a temporary directory that will be automatically cleaned up
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    # Create temporary file in the temporary directory
                    temp_pdf_path = os.path.join(temp_dir, uploaded_file.name)
                    try:
                        with open(temp_pdf_path, "wb") as file:
                            file.write(uploaded_file.getvalue())
                        loader = PyPDFLoader(temp_pdf_path)
                        docs = loader.load()
                        documents.extend(docs)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue
                    # No need to manually remove files - TemporaryDirectory handles cleanup

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
            splits = text_splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(splits, embeddings)
            vectorstore.save_local("faiss_index")
            retriever = vectorstore.as_retriever()

            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            try:
                history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
            except Exception as e:
                st.error(f"Error creating history-aware retriever: {str(e)}")
                st.stop()

            system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise.\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]


            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            # Display chat history
            session_history = get_session_history(session_id)
            display_chat_messages(session_history)

            # Initialize input state if not present
            if 'user_input' not in st.session_state:
                st.session_state.user_input = ''

            # Input field for the user's question
            user_input = st.text_input("Your question:", key="question_input", value=st.session_state.user_input)

            # Submit button
            if st.button("Send"):
                if user_input.strip():  # Only process non-empty input
                    try:
                        # Clear the input field by updating session state
                        st.session_state.user_input = ''

                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}},
                        )
                        if response and "answer" in response:
                            # No need to display here as it's handled by display_chat_messages
                            st.rerun()
                        else:
                            st.markdown("**Assistant:** No answer received.")
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
