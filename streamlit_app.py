import streamlit as st
import time
from typing import Generator
from langchain_google_genai import ChatGoogleGenerativeAI

def display_chat_history():
    """Displays chat messages from history on app rerun."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

@st.fragment
def response_generator(response: str) -> Generator[str, None, None]:
    """Generates a stream of words from a response, handling None values."""
    if not response:
        yield "I'm sorry, there was no response."
        return
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

# Function to add a turn to the context
def add_to_context(user_input, ai_response):
    st.session_state.context.append({"user": user_input, "ai": ai_response})

# Function to generate the full prompt from context
def generate_full_prompt(prompt):
    full_prompt = ""
    for turn in st.session_state.context[-5:]:  # Keep the last 5 turns to limit token usage
        full_prompt += f"User: {turn['user']}\nAI: {turn['ai']}\n"
    full_prompt += f"User: {prompt}"
    return full_prompt

def ai_stream_response(respond):
    with st.chat_message("assistant"):
        st.write_stream(response_generator(respond))

@st.cache_resource
def connect_llm(geminiapikey):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25",api_key=geminiapikey)
    return llm

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = []
if "geminiapikey" not in st.session_state:
    st.session_state.geminiapikey = ''


st.sidebar.html("<center><h1>TASKY</h1></center>")
st.sidebar.html("<center>Time-Saving Assistive Smart and Knowledgeable as Your Companion  </center>")
st.sidebar.html("<center>Developed Using <br><a href='https://github.com/browser-use/browser-use'><img src='https://browser-use.com/logo.png' alt='Browser Use Logo' width='30' height='30'></a><a href='https://github.com/streamlit/streamlit'><img src='https://docs.streamlit.io/logo.svg' alt='Streamlit Logo' width='40' height='40'></a></center>")
st.session_state.geminiapikey = st.sidebar.text_input("GEMINI API Key", type="password")
st.sidebar.html("<h3>Your api key is not saved and only used for your current session.</h3>")
#respond = "This is my only response to you"

if st.session_state.geminiapikey!='':
    llm = connect_llm(st.session_state.geminiapikey)

display_chat_history()
prompt = st.chat_input("Say something")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # generate full prompt that include the context
    full_prompt = generate_full_prompt(prompt+".In your response don't add AI:")
    #
    ai_msg = llm.invoke(full_prompt)
    respond = ai_msg.content
    ai_stream_response(respond)
    st.session_state.messages.append({"role": "assistant", "content": respond})
    add_to_context(prompt, respond)
    st.rerun()
    _='''
    with st.chat_message("assistant"):
        st.write_stream(st.session_state.context)'
    '''