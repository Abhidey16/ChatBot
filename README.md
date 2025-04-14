# DeepSeek Chat Assistant

![DeepSeek Chat Assistant](https://i.imgur.com/your-image-url.png)

## Overview

DeepSeek Chat Assistant is an interactive conversational AI application built with Streamlit that leverages the DeepSeek-R1-Distill-Qwen-1.5B model. This application provides a user-friendly chat interface for interacting with a state-of-the-art language model capable of generating human-like responses to a wide range of queries.

## Features

- **Interactive Chat Interface**: Clean, intuitive chat-style UI for seamless interaction
- **Response Timing**: Displays how long each response took to generate
- **Adjustable Parameters**: Control the model's creativity and response style with temperature and top-p settings
- **Chat History**: Maintains conversation history within a session
- **GPU Acceleration**: Optimized for hardware acceleration using CUDA
- **Memory Efficient**: Uses half-precision (FP16) for reduced memory footprint

## Requirements

### Hardware Requirements
- NVIDIA GPU with at least 4GB VRAM recommended
- 8GB+ system RAM
- Internet connection for initial model download

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA drivers (for GPU acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepseek-chat-assistant.git
cd deepseek-chat-assistant
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install streamlit torch transformers accelerate
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. The application will open in your web browser at `http://localhost:8501`

3. Wait for the model to load (this may take a few moments on first run)

4. Type your message in the input box at the bottom of the screen and press Enter

5. Adjust generation parameters in the sidebar:
   - **Temperature**: Controls randomness (higher values = more creative, diverse responses)
   - **Top P**: Controls response diversity by limiting token selection to the top probability mass

6. Use the "Clear Chat History" button to start a fresh conversation

## How It Works

### Model Information

The application uses DeepSeek-R1-Distill-Qwen-1.5B, a distilled version of the larger Qwen model optimized for efficiency. This model offers a good balance between performance and resource requirements, making it suitable for running on consumer hardware.

### Code Structure

- **Model Loading**: Uses Streamlit's caching to prevent reloading the model with each interaction
- **Response Generation**: Formats prompts using a chat template and generates responses with configurable parameters
- **Response Timing**: Tracks and displays how long each response takes to generate
- **Session Management**: Uses Streamlit's session state to maintain conversation history

### Response Generation Process

1. User input is formatted as a chat message
2. The input is tokenized and processed by the model
3. The model generates a response based on temperature and top-p parameters
4. Response time is calculated and displayed
5. Both the response and timing information are saved to the chat history

## Customization

### Changing the Model

You can modify the `model_name` variable in the `load_model()` function to use a different model:

```python
model_name = "another-model/model-name"
```

### Adjusting Generation Parameters

Default parameters can be modified:
- `max_new_tokens`: Controls maximum response length (default: 1024)
- Default temperature and top-p values in the slider definitions

### Styling

The application uses Streamlit's default styling, but you can customize it by:
1. Creating a `.streamlit/config.toml` file with custom theme settings
2. Adding custom CSS with `st.markdown("""<style>...</style>""", unsafe_allow_html=True)`

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**: Reduce `max_new_tokens` or try a smaller model
2. **CUDA Not Available**: Ensure CUDA is properly installed and compatible with PyTorch
3. **Slow Responses**: Reduce `max_new_tokens` or adjust temperature/top-p for faster generation

### Model Loading Issues

If you encounter issues loading the model:
```bash
pip install huggingface_hub
huggingface-cli login
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DeepSeek AI](https://github.com/deepseek-ai) for providing the open-source model
- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Streamlit](https://streamlit.io/) for the web application framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Created by **Abhi Dey**
