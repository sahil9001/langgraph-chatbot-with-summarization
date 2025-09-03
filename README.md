# LangGraph Chatbot with Chainlit Interface

This project implements a conversational AI chatbot using LangGraph for conversation management and Chainlit for the web interface. The chatbot features automatic conversation summarization to maintain context while keeping memory usage efficient.

## Features

- **LangGraph Workflow**: Uses LangGraph for managing conversation state and flow
- **Automatic Summarization**: Automatically summarizes conversations when they exceed 6 messages
- **Memory Management**: Efficiently manages conversation history using checkpoints
- **Modern Web Interface**: Beautiful Chainlit-based chat interface
- **Ollama Integration**: Uses local Ollama models (phi3:mini by default)

## Architecture

The system consists of two main components:

1. **`app.py`**: Contains the core LangGraph workflow logic
   - Conversation state management
   - Automatic summarization
   - Message processing pipeline

2. **`chainlit.py`**: Provides the web interface
   - User-friendly chat interface
   - Session management
   - Integration with the LangGraph workflow

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- phi3:mini model downloaded (`ollama pull phi3:mini`)

## Installation

1. Clone the repository and navigate to the project directory:
```bash
cd chatbot_with_summarization
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure Ollama is running and the model is available:
```bash
ollama serve
ollama pull phi3:mini
```

## Usage

### Running the Chainlit Interface

Start the Chainlit chatbot:
```bash
chainlit run chainlit.py
```

The chatbot will be available at `http://localhost:8000`

### Testing the Core Logic

You can also test the core LangGraph workflow directly:
```bash
python app.py
```

## How It Works

1. **Conversation Start**: When a user starts chatting, a unique thread ID is created
2. **Message Processing**: Each message is processed through the LangGraph workflow
3. **Automatic Summarization**: When the conversation exceeds 6 messages, the system automatically summarizes the conversation
4. **Memory Cleanup**: Old messages are removed, keeping only the most recent ones
5. **Context Preservation**: The summary maintains conversation context for future interactions

## Configuration

### Model Settings

You can modify the model settings in `app.py`:
```python
def create_model():
    return ChatOllama(
        model="phi3:mini",  # Change model here
        temperature=0.8,     # Adjust creativity
        num_predict=256,     # Adjust response length
    )
```

### Summarization Threshold

Adjust when summarization occurs by modifying the `should_continue` function:
```python
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6:  # Change this number
        return "summarize_conversation"
    return END
```

### Chainlit UI

Customize the interface appearance in `.chainlit/config.toml`:
- Theme (light/dark)
- Assistant name and description
- Chat features and settings

## Project Structure

```
chatbot_with_summarization/
├── app.py                 # Core LangGraph workflow
├── chainlit.py           # Chainlit web interface
├── requirements.txt      # Python dependencies
├── .chainlit/           # Chainlit configuration
│   └── config.toml
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**: Ensure Ollama is running (`ollama serve`)
2. **Model Not Found**: Download the required model (`ollama pull phi3:mini`)
3. **Port Already in Use**: Chainlit uses port 8000 by default. Change it with `chainlit run chainlit.py --port 8001`

### Performance Tips

- Use smaller models for faster responses
- Adjust the summarization threshold based on your needs
- Monitor memory usage with very long conversations

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
