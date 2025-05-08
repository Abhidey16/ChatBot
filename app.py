import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import time
import google.generativeai as genai
from openai import OpenAI
import json
from pathlib import Path
import requests, os

# Define path to config directory and file
config_dir = Path("config")
config_file = "api_keys.json"

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function for the New Chat button
def start_new_chat():
    st.session_state.chat_history = []

# Title and description directly in the main area
st.title("AI ChatBot")
st.write("Generate creative text using AI models")

# Function to load API keys from config file
def load_api_keys():
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return json.load(f)
    return {"gemini": "", "grok": "", "GPT-4": ""}

# Load API keys from the config file
api_keys = load_api_keys()
gemini_key = api_keys.get("gemini", "")
grok_key = api_keys.get("grok", "")
GPT_key = api_keys.get("GPT-4", "")

# Model selection in sidebar (without showing API keys)
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.radio("Choose AI Model", ["Gemini (gemini-2.0-flash)", "Grok", "GPT-4"])

# Add generation parameters in the sidebar
st.sidebar.header("Generation Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
                              help="Higher values make output more random, lower values make it more deterministic")
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                        help="Controls diversity via nucleus sampling: 1.0 considers all tokens, lower values restrict to more likely tokens")

# Add New Chat button to the sidebar, below the parameters
st.sidebar.button("New Chat", on_click=start_new_chat, key="new_chat_btn")

# Function to generate response using Gemini API
def generate_response_with_gemini(prompt, temperature=0.7, top_p=0.9):
    start_time = time.time()
    
    # Configure the Gemini API with key from config file
    genai.configure(api_key=gemini_key)
    
    # Use the gemini-2.0-flash model with specified parameters
    model = genai.GenerativeModel('gemini-2.0-flash',
                                 generation_config={"temperature": temperature, "top_p": top_p})
    
    # Generate the response
    response = model.generate_content(prompt)
    
    elapsed_time = time.time() - start_time
    return response.text, elapsed_time, "Gemini (gemini-2.0-flash)"

# Create a function to generate response using GPT-4
def generate_response_with_gpt4(prompt, temperature=0.7, top_p=0.9):
    start_time = time.time()
    
    # Create OpenAI client with your API key
    client = OpenAI(api_key=GPT_key)
    
    # Generate the response using GPT-4 with specified parameters
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates creative and informative text."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=1000
    )
    
    elapsed_time = time.time() - start_time
    return response.choices[0].message.content, elapsed_time, "GPT-4"

# Function to generate response using Grok API
def generate_response_with_grok(prompt, temperature=0.7, top_p=0.9):
    start_time = time.time()
    
    # Grok API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Headers for the API request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok_key}"
    }
    
    # Prepare the request payload with specified parameters
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that generates creative and informative text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": 2024
    }
    
    # Make the API request
    response = requests.post(url, headers=headers, json=payload)
    
    # Process the response
    if response.status_code == 200:
        result = response.json()
        answer = result.get("choices", [{}])[0].get("message", {}).get("content", "No response generated")
    else:
        answer = f"Error: Received status code {response.status_code} from Grok API. {response.text}"
    
    elapsed_time = time.time() - start_time
    return answer, elapsed_time, "Grok"

# Display existing chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"### You")
        st.write(message["content"])
    else:
        # Use the model name stored in the message, not the current selected model
        st.markdown(f"### {message.get('model', 'AI')} Response")
        st.write(message["content"])
        if "time" in message:
            st.caption(f"Response time: {message['time']:.2f} seconds")
        # Display generation parameters if available
        if "temperature" in message and "top_p" in message:
            st.caption(f"Parameters: Temperature={message['temperature']}, Top P={message['top_p']}")
        st.divider()  # Add separator between prompt-response pairs

# Using chat_input instead of text_input
if prompt := st.chat_input("Enter your text generation prompt:"):
    # Add prompt to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display the new prompt
    st.markdown("### You")
    st.write(prompt)

    # Check if API key is available
    if (model_choice == "Gemini (gemini-2.0-flash)" and not gemini_key) or (model_choice == "Grok" and not grok_key) or (model_choice == "GPT-4" and not GPT_key):
        st.error(f"API key for {model_choice} not found in config file. Please add it to {config_file}")
    else:
        with st.spinner(f"Generating text using {model_choice}..."):
            try:
                # Generate response based on selected model with temperature and top_p parameters
                if model_choice == "Gemini (gemini-2.0-flash)":
                    answer, elapsed_time, model_name = generate_response_with_gemini(prompt, temperature, top_p)
                elif model_choice == "GPT-4":
                    answer, elapsed_time, model_name = generate_response_with_gpt4(prompt, temperature, top_p)
                elif model_choice == "Grok":
                    answer, elapsed_time, model_name = generate_response_with_grok(prompt, temperature, top_p)
                else:
                    st.error("Unknown model selected")
                    st.stop()
                
                # Add answer to chat history with the model name and generation parameters
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": answer, 
                    "time": elapsed_time,
                    "model": model_name,  # Store the model name used for this response
                    "temperature": temperature,  # Store temperature used
                    "top_p": top_p  # Store top_p used
                })
                
                # Display answer and response time
                st.markdown(f"### {model_name} Response")
                st.write(answer)
                st.caption(f"Response time: {elapsed_time:.2f} seconds")
                st.caption(f"Parameters: Temperature={temperature}, Top P={top_p}")
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": error_message,
                    "model": model_choice,  # Store the model that was attempted
                    "temperature": temperature,
                    "top_p": top_p
                })
