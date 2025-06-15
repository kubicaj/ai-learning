import uuid
import json
import gradio as gr

from typing import Annotated

from IPython.display import Image, display
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from pydantic import BaseModel

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from src.langgraph.llm.llm_factory import LLMFactory
from src.openai.common.logger import init_logger


# //////////////// Step 1: Create state ////////////////

class ChatState(BaseModel):
    """
    Chat of LangGraph Chat APP
    """

    # list of messages.
    # Use `add_messages` reducer which Merges two lists of messages, updating existing messages by ID
    # By default, this ensures the state is "append-only", unless the
    # new message has the same ID as an existing message.
    messages: Annotated[list, add_messages]


# //////////////// Step 2: Start the Graph Builder with this State class ////////////////

graph_builder = StateGraph(ChatState)


# //////////////// Step 3: Create a Node ////////////////

def chatbot_node(old_state: ChatState) -> ChatState:
    response = LLMFactory.get_chat_open_ai_llm().invoke(old_state.messages)
    new_state = ChatState(messages=[response])
    return new_state


# add note into builder
graph_builder.add_node("chatbot_node", chatbot_node)

# //////////////// Step 4: Create Edges ////////////////

graph_builder.add_edge(START, "chatbot_node")
graph_builder.add_edge("chatbot_node", END)

# //////////////// Step 5: Compile the Graph ////////////////

# you can see that memory is added there
graph = graph_builder.compile(checkpointer=MemorySaver())
display(Image(graph.get_graph().draw_mermaid_png()))

# //////////////// Step 6: Create simple Chat UI ////////////////

logger = init_logger()


def start_session():
    """
    Create ID for conversation
    """
    return str(uuid.uuid4())


def chat(user_input: str, history, session_id: str):
    """
    Method which is invoked each time after user question

    Args:
        user_input - input query from user
        history - chat history (not used in this case because the memory of langgraph will be used
    """
    logger.info(f"[{session_id}] New message: {user_input}")
    initial_state = ChatState(messages=[{"role": "user", "content": user_input}])

    # config is needed to identify session_id/thread_id which is checkpointing the history of conversation
    result = graph.invoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": session_id
            }
        }
    )
    logger.info(result)
    return result['messages'][-1].content


gr.ChatInterface(
    chat,
    type="messages",
    additional_inputs=[
        gr.State(start_session())
    ],
).launch()
