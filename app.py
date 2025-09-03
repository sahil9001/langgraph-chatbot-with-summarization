from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, END
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from typing_extensions import Literal
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START


model = ChatOllama(
    model = "phi3:mini",
    validate_model_on_init = True,
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)

class State(MessagesState): 
    summary: str


def call_model(state: State):
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of the conversation: {summary}"
        messages = [SystemMessage(content=system_message)] + state['messages']
    else:
        messages = state["messages"]

    response = model.invoke(messages)
    return {"messages": response}

def summarize_conversation(state: State):
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


def should_continue(state: State) -> Literal ["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6: 
        return "summarize_conversation"
    return END


# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# Create a thread
config = {"configurable": {"thread_id": "1"}}

# Start conversation
input_message = HumanMessage(content="hi! I'm Lance")
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