import streamlit as st
import asyncio
import logging
import time
from typing import Generator, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from browser_use import Agent, Browser, BrowserConfig
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor # Consider using ThreadPoolExecutor for Browser Use

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
def connect_llm(apikey):
    llm = ChatOpenAI(model="gpt-4o",api_key=apikey)
    #llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25",api_key=apikey)
    #llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=SecretStr(os.getenv('GEMINI_API_KEY')).get_secret_value())
    return llm

# Function to run the agent (asynchronous)
async def run_agent(task, llm,browser):
    try:
        agent = Agent(task=task, llm=llm, use_vision=False,browser=browser)
        
        result = await agent.run()  # Await the agent's execution
        return result
    except Exception as e:
        logging.error(f"Error occurred while running the agent: {e}")
        st.error(f"Error occurred while running the agent: {e}") #Display Error message so user knows something went wrong.
        return None

# Process prompt function, this function is outside the main function to allow process isolation.
def process_prompt(prompt: str, llm: any,browser:any) -> Tuple[str, str, str, str]:
    """Processes a prompt using the language model and agent, handling errors."""
    try:
        if llm is None:
            logging.error("Failed to initialize LLM in process_prompt")
            return "I'm sorry, the language model failed to initialize.", "", "", "" # Return empty strings to prevent downstream errors.

        agent_result = asyncio.run(run_agent(prompt, llm,browser))  # Run the agent and get the response

        if agent_result is None: # Handle the case when the agent fails.
            return "I'm sorry, the agent returned no result.", "", "", ""
        
        final = agent_result.final_result()
        thoughts = agent_result.model_thoughts()
        extracted = agent_result.extracted_content()

        return (
            str(agent_result),
            str(final) if final else "", #Return empty string if final is None to avoid error.
            str(thoughts) if thoughts else "",
            str(extracted) if extracted else "",
        )

    except Exception as e:
        logging.error(f"An error occurred in process_prompt: {e}")
        return f"I'm sorry, an error occurred: {e}", "", "", ""

# Basic configuration
config = BrowserConfig(
    headless=True,
    disable_security=True
)

browser = Browser(config=config)

# executor = ProcessPoolExecutor(max_workers=1) #Switching back to ThreadPoolExecutor due to issues with browser_use and ProcessPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = []
if "apikey" not in st.session_state:
    st.session_state.apikey = ''


def main():
    st.sidebar.html("<center><h1>TASKY</h1></center>")
    st.sidebar.html("<center>Time-Saving Assistive Smart and Knowledgeable as Your Companion  </center>")
    st.sidebar.html("<center>Developed Using <br><a href='https://github.com/browser-use/browser-use'><img src='https://browser-use.com/logo.png' alt='Browser Use Logo' width='30' height='30'></a><a href='https://github.com/streamlit/streamlit'><img src='https://docs.streamlit.io/logo.svg' alt='Streamlit Logo' width='40' height='40'></a></center>")
    st.session_state.apikey = st.sidebar.text_input("OpenAI API Key", type="password")
    st.sidebar.html("<h3>Your api key is not saved and only used for your current session.</h3>")

    if st.session_state.apikey: #Use If statement rather than if != ''
        try:
            llm = connect_llm(st.session_state.apikey)
        except Exception as e:
            st.error(f"Failed to connect to the language model: {e}")
            llm = None  # Ensure llm is None if connection fails
    else:
        llm = None
        st.warning("Please enter your OpenAI API key.")

    display_chat_history()

    prompt = st.chat_input("Say something")
    if prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if llm: # Only Proceed if llm is connected.
            # generate full prompt that include the context
            full_prompt = generate_full_prompt(prompt + ".In your response don't add AI:")

            # Initial AI response
            try:
                ai_msg = llm.invoke(full_prompt)
                respond = ai_msg.content
                ai_stream_response(respond)
                st.session_state.messages.append({"role": "assistant", "content": respond})
                add_to_context(prompt, respond)
            except Exception as e:
                respond = f"Error generating initial response: {e}"
                st.error(respond)
                st.session_state.messages.append({"role": "assistant", "content": respond})


            # Run Agent
            future = executor.submit(process_prompt, prompt, llm, browser)
            assistant_response, final_result, model_thoughts, extracted = future.result()

            ai_stream_response(final_result)
            st.session_state.messages.append({"role": "assistant", "content": final_result})
            add_to_context(prompt, final_result)
        else:
            st.warning("Please enter a valid OpenAI API Key to get a response.")
            final_result = "Please enter a valid OpenAI API Key to get a response."

        st.rerun()


# Run the Streamlit app
if __name__ == "__main__":
    main()