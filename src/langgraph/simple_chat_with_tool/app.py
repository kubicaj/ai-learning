import uuid
import gradio as gr

from typing import Annotated

from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
from IPython.display import Image, display
from dotenv import load_dotenv
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from src.langgraph.llm.llm_factory import LLMFactory
from src.openai.common.logger import init_logger

import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# get to load environment variables
load_dotenv(override=True)

# create google search tool
serper = GoogleSerperAPIWrapper()
tool_search = Tool(
    name="search",
    func=serper.run,
    description="Useful for when you need more information from an online search"
)
tools_list = [tool_search]

# create llm
open_ai_llm = LLMFactory.get_chat_open_ai_llm()
llm_with_tools = open_ai_llm.bind_tools(tools_list)

# create memory - only using internal memory object valid per python process
llm_memory = MemorySaver()

# use sql memory if you want
db_path = "chatbot_memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)

# flag what memory to use
used_memory = sql_memory


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


# //////////////// Step 3: Create a Nodes ////////////////

def chatbot_node(old_state: ChatState) -> ChatState:
    response = llm_with_tools.invoke(old_state.messages)
    new_state = ChatState(messages=[response])
    return new_state


# add note into builder
graph_builder.add_node("chatbot_node", chatbot_node)
# consider tools as a node
graph_builder.add_node("tools", ToolNode(tools=tools_list))

# //////////////// Step 4: Create Edges ////////////////

# conditionally run tools if needed
graph_builder.add_conditional_edges("chatbot_node", tools_condition, "tools")
# this will help to loop around
graph_builder.add_edge("tools", "chatbot_node")
graph_builder.add_edge(START, "chatbot_node")

# //////////////// Step 5: Compile the Graph ////////////////

# you can see that memory is added there
graph = graph_builder.compile(checkpointer=sql_memory)
display(Image(graph.get_graph().draw_mermaid_png()))

# //////////////// Step 6: Create simple Chat UI ////////////////

logger = init_logger()


def start_session():
    """
    Create ID for conversation
    """
    if isinstance(used_memory, SqliteSaver):
        # use the same session ID each time in case of long term SQL memory
        return 1
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
