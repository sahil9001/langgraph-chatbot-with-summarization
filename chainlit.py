import chainlit as cl
from app import create_workflow, get_or_create_thread, HumanMessage
import uuid


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Create a unique thread ID for this session
    session_id = str(uuid.uuid4())
    cl.user_session.set("thread_id", session_id)
    
    # Initialize the workflow
    cl.user_session.set("workflow", create_workflow())
    
    # Welcome message
    await cl.Message(
        content="👋 Hello! I'm your AI assistant powered by LangGraph. I can remember our conversation and summarize it when it gets long. How can I help you today?",
        author="Assistant"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    try:
        # Get the thread ID and workflow from the session
        thread_id = cl.user_session.get("thread_id")
        workflow = cl.user_session.get("workflow")
        
        # Get thread configuration
        config = get_or_create_thread(thread_id)
        
        # Create the input message
        input_message = HumanMessage(content=message.content)
        
        # Process through the workflow
        output = workflow.invoke({"messages": [input_message]}, config)
        
        # Get the response
        if output['messages']:
            response_content = output['messages'][-1].content
            
            # Check if summarization happened
            if 'summary' in output and output['summary']:
                # Send a notification about summarization
                await cl.Message(
                    content=f"💡 I've summarized our conversation to keep it concise. Current summary: {output['summary'][:100]}...",
                    author="System"
                ).send()
            
            # Send the main response
            await cl.Message(
                content=response_content,
                author="Assistant"
            ).send()
        else:
            await cl.Message(
                content="I apologize, but I couldn't generate a response. Please try again.",
                author="Assistant"
            ).send()
            
    except Exception as e:
        # Handle errors gracefully
        await cl.Message(
            content=f"An error occurred: {str(e)}. Please try again.",
            author="System"
        ).send()


@cl.on_chat_end
async def end():
    """Clean up when chat ends"""
    # Clear session data
    cl.user_session.clear()
