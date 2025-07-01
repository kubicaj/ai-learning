from abc import ABC, abstractmethod
from typing import Any, List

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.tools.google_search import tool_search_for_interview
from src.openai.common.logger import init_logger


class InterviewAgent(ABC):
    """
    Abstract class which is used by all interview agents
    """

    def __init__(self):
        self.logger = init_logger()

    @abstractmethod
    def agent_callback(self, graph_state: GraphState) -> Any:
        """
        Agent callback which is adding into langraph
        """
        pass

    @staticmethod
    def get_tools() -> List[Any]:
        return [
            tool_search_for_interview
        ]

    def call_as_standalone(self, initial_state: GraphState):
        """
        Call the agent as a standalone app.
        It can be used for testing/debuting purpose
        """
        # //////////////// First Initialization ////////////////

        graph_builder = StateGraph(GraphState)

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
        graph = graph_builder.compile()

        # //////////////// Invoke the Graph ////////////////
        result = graph.invoke(initial_state)
        return result
