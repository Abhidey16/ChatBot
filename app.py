import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="DeepSeek Chat App",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_model():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def generate_response(prompt, tokenizer, model, temperature, top_p):
    # Format the prompt for chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=512,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clean response if needed
    if "</think>" in response:
        cleaned_response = response.split("</think>")[1]
        return cleaned_response.strip()
    return response

def main():
    st.title("🤖 DeepSeek Chat Assistant")
    
    st.markdown("""
    Welcome to the DeepSeek Chat Assistant! This app uses the DeepSeek-R1-Distill-Qwen-1.5B model 
    to generate responses to your queries.
    """)
    
    # Add sidebar with sliders for temperature and top_p
    with st.sidebar:
        st.header("Generation Parameters")
        temperature = st.slider(
            "Temperature", 
            min_value=0.1, 
            max_value=1.5, 
            value=0.5, 
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        top_p = st.slider(
            "Top P", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Limits token selection to a cumulative probability threshold"
        )
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Load model (will use cached version after first load)
    with st.spinner("Loading model... (this may take a moment)"):
        tokenizer, model = load_model()
    
    # Chat input
    if prompt := st.chat_input("What would you like to ask?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, tokenizer, model, temperature, top_p)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
    # Add a button to clear chat history
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
