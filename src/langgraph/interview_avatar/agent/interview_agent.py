import inspect
import os
import sqlite3
from pypdf import PdfReader
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.tools.google_search import tool_search_for_interview
from src.openai.common.logger import init_logger


def _load_md_resource_file(file_name: str) -> str:
    """
    Load position description

    Return:
        str representation of position desc
    """
    with open(f'../../resources/{file_name}', 'r') as file:
        return file.read()

def get_pdf_content(pdf_path: str) -> str:
    """
    Read PDF

    Args:
        - pdf_path: Path to PDF

    Returns:
        - text content of PDF
    """
    reader = PdfReader(pdf_path)
    pdf_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text
    return pdf_text

CANDIDATE_CV = get_pdf_content("../../resources/cv_candidate.pdf")
POSITION_DESCRIPTION = _load_md_resource_file("position_description.md")
TOPICS_TO_INTERVIEW = _load_md_resource_file("technical_topics_to_interview.md")

class InterviewAgent(ABC):
    """
    Abstract class which is used by all interview agents
    """

    def __init__(self):
        self.logger = init_logger()
        self.agent_prompt_templates = self.load_agent_prompt()

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

    @staticmethod
    def get_prompt_names() -> list[str]:
        """
        Get name of prompt files which are relevant for the Agent
        """
        # default is agent prompt
        return ["agent_prompt"]

    def load_agent_prompt(self, placeholders: dict[str, str] = None) -> dict[str, str] :
        """
        Loads agent prompt from default location resources/agent_prompt.md

        Args:
            placeholders dict[str, str]: Placeholders in prompt

        Returns:
            str: Content of the Markdown file.
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
                result[prompt_file_name] = result[prompt_file_name] .format(**placeholders)
        return result

    def call_as_standalone(self, initial_state: GraphState, memory_id: str = None,
                           compiled_state_graph: StateGraph = None) -> Tuple[
        Any, StateGraph]:
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
