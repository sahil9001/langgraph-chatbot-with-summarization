from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from typing_extensions import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START


def create_model():
    """Create and return the ChatOllama model instance"""
    return ChatOllama(
        model="phi3:mini",
        validate_model_on_init=True,
        temperature=0.8,
        num_predict=256,
    )


model = create_model()


class State(MessagesState): 
    summary: str


def call_model(state: State):
    """Call the model with the current state"""
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of the conversation: {summary}"
        messages = [SystemMessage(content=system_message)] + state['messages']
    else:
        messages = state["messages"]

    response = model.invoke(messages)
    return {"messages": response}


def summarize_conversation(state: State):
    """Summarize the conversation and clean up old messages"""
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but 2 messages (which are most recent)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def should_continue(state: State) -> Literal["summarize_conversation", END]:
    """Determine whether to continue or summarize"""
    messages = state["messages"]
    if len(messages) > 6: 
        return "summarize_conversation"
    return END


def create_workflow():
    """Create and return the compiled workflow graph"""
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node("summarize_conversation", summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def get_or_create_thread(thread_id: str):
    """Get or create a thread configuration"""
    return {"configurable": {"thread_id": thread_id}}


def process_message(message: str, thread_id: str = "default"):
    """Process a single message through the workflow"""
    graph = create_workflow()
    config = get_or_create_thread(thread_id)
    
    input_message = HumanMessage(content=message)
    output = graph.invoke({"messages": [input_message]}, config)
    
    # Return the last message content
    if output['messages']:
        return output['messages'][-1].content
    return "No response generated"


# Create the graph object for LangGraph CLI compatibility
graph = create_workflow()


# Example usage (for testing)
if __name__ == "__main__":
    # Start conversation
    input_message = HumanMessage(content="hi! I'm Lance")
    config = get_or_create_thread("1")
    
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="what's my name?")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-1:]:
        m.pretty_print()

    input_message = HumanMessage(content="i like the 49ers!")
    output = graph.invoke({"messages": [input_message]}, config) 
    for m in output['messages'][-1:]:
        m.pretty_print()