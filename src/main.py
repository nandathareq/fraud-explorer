import streamlit as st

from util.llm import init_llm, app
from config import LLM_ENGINE
from util.data import init_data_pipeline
import requests

def initialize(url):
    st.session_state.llm, st.session_state.embedding = init_llm(url)
    st.session_state.db_path, st.session_state.vectordb = init_data_pipeline()         
    st.rerun()

# --- UI Setup ---
st.title("APP_TITLE")


@st.dialog("Input LLM Engine URL")
def show_pop_up():
    st.write(f"open and run this collab { LLM_ENGINE}")
    url = st.text_input("paste ngrok url here ..")
    if st.button("Submit"):
        initialize(url)
        
        
# --- Utility Functions ---
if "llm" not in st.session_state:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        initialize(None)
    except requests.exceptions.RequestException:
        show_pop_up()


st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        result = app.invoke({"question": prompt})["result"]

        full_answer = []  # list, bukan string

        def collect_stream():
            for chunk in result["stream"]:
                full_answer.append(chunk.content)
                yield chunk

        if result.get("references"):
            st.markdown(result["references"])
        st.write_stream(collect_stream())

    st.session_state.messages.append({
        "role": "assistant",
        "content": "".join(full_answer)
    })