import inspect
import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from langchain_core.messages import ToolMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph.interview_avatar.interview_config import InterviewConfig
from src.langgraph.interview_avatar.pojo.interview_graph_state import InterviewGraphState
from src.langgraph.interview_avatar.tools.google_search import tool_search_for_interview
from src.openai.common.logger import init_logger


class InterviewAgent(ABC):
    """
    Abstract class which is used by all interview agents
    """

    MAX_ALLOWED_ITERATIONS = 10

    def __init__(self, chosen_position: str = None):
        self.logger = init_logger()
        self.agent_prompt_templates = self.load_agent_prompt()
        self._interview_config = InterviewConfig.get_active_instance()
        self.chosen_position = chosen_position

    @abstractmethod
    def agent_callback_implementation(self, graph_state: InterviewGraphState) -> Any:
        """
        Agent callback which is adding into LangGraph
        """
        pass

    @staticmethod
    def get_tools() -> List[Any]:
        """
        Get list of available tools

        Return:
            list of tools
        """
        return [
            tool_search_for_interview
        ]

    @staticmethod
    def get_prompt_names() -> list[str]:
        """
        Get name of prompt files which are relevant for the Agent
        """
        # default is agent prompt
        return ["agent_prompt"]

    def agent_callback(self, graph_state: InterviewGraphState) -> Any:
        """
        General callback function

        Args:
            graph_state - graph state

        Return:
            return value from callback. Typically, it will be new state
        """
        self.logger.info(f"Invoking agent: {self.__class__.__name__} with state \n {graph_state}")
        if graph_state.agent_iterations > self.MAX_ALLOWED_ITERATIONS:
            raise Exception("The maximal iterations has been reached. Try again")
        result = self.agent_callback_implementation(graph_state)
        self.logger.info(f"Result from agent: {self.__class__.__name__} = \n {result}")
        return result

    def process_tool_response(self, response: BaseMessage, messages: List[BaseMessage],
                              llm_client: ChatOpenAI) -> BaseMessage:
        """
        Process response with tools

        Args:
            response - base response wil tool calls
            messages - history of messages
            llm_client - llm API client

        Args:
            response from llm
        """
        self.logger.debug("Tools was found to call. Going to process it")
        tool_messages = []
        tools = self.get_tools()
        for tool_call in response.tool_calls:
            # loop all tools and create final tool message
            tool_name = tool_call["name"]
            # Find and invoke the tool
            tool_fn = next((t for t in tools if t.name == tool_name), None)
            if tool_fn is None:
                raise ValueError(f"Tool {tool_name} not found.")

            tool_result = tool_fn.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(
                tool_call_id=tool_call["id"],
                content=tool_result
            ))

        # Re-invoke the model with the tool response
        all_messages = messages + [response] + tool_messages
        return llm_client.invoke(input=all_messages)

    def load_agent_prompt(self, placeholders: dict[str, str] = None) -> dict[str, str]:
        """
        Loads agent prompt from default location resources/agent_prompt.md

        Args:
            placeholders dict[str, str]: Placeholders in prompt

        Returns:
            dict: key = MD file name and value = Content of the Markdown file.
        """
        file_path = inspect.getfile(self.__class__)
        class_file_location = os.path.dirname(os.path.abspath(file_path))
        agent_prompt_files = self.get_prompt_names()
        result: dict[str, str] = {}
        for prompt_file_name in agent_prompt_files:
            with open(f"{class_file_location}{os.path.sep}resources{os.path.sep}{prompt_file_name}.md", 'r',
                      encoding='utf-8') as file:
                result[prompt_file_name] = file.read()
            if placeholders:
                result[prompt_file_name] = result[prompt_file_name].format(**placeholders)
        return result

    def call_as_standalone(self, initial_state: InterviewGraphState, memory_id: str = None,
                           compiled_state_graph: StateGraph = None) -> Tuple[
        Any, StateGraph]:
        """
        Call the agent as a standalone app.
        It can be used for testing/debuting purpose
        """
        # //////////////// First Initialization ////////////////

        graph_builder = StateGraph(InterviewGraphState)

        # //////////////// Create Nodes ////////////////

        # add note into builder
        graph_builder.add_node("simple_node", self.agent_callback)
        # consider tools as a node
        graph_builder.add_node("tools", ToolNode(tools=self.get_tools()))

        # //////////////// Create Edges ////////////////

        # conditionally run tools if needed
        graph_builder.add_conditional_edges("simple_node", tools_condition, "tools")
        # this will help to loop around
        graph_builder.add_edge("tools", "simple_node")
        graph_builder.add_edge(START, "simple_node")

        # //////////////// Compile the Graph ////////////////

        # Compile the graph
        graph_config = {}
        if not compiled_state_graph:
            if memory_id:
                compiled_state_graph = graph_builder.compile(checkpointer=MemorySaver())
            else:
                compiled_state_graph = graph_builder.compile()

        if memory_id:
            graph_config = {
                "configurable": {
                    "thread_id": memory_id
                }
            }
        # //////////////// Create memory if memory id is setup ////////////////

        # //////////////// Invoke the Graph ////////////////
        result = compiled_state_graph.invoke(
            initial_state,
            config=graph_config
        )
        return result, compiled_state_graph
