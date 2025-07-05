import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.agent.interview_manager.interview_manager import InterviewManager
from src.langgraph.interview_avatar.agent.interview_question_evaluator.interview_question_evaluator import \
    InterviewQuestionEvaluator
from src.langgraph.interview_avatar.agent.interview_question_generator.interview_question_generator import \
    InterviewQuestionGenerator
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.openai.common.logger import init_logger


class InterviewOrchestration:
    """
    Class represent the interview orchestration of agents
    It will create LangGraph implementation of agent flows together with conditions
    """
    TOOLS_AGENT_NAME = "tools"

    def __init__(self):
        # here init dict of all nodes
        self.nodes_dict = {
            InterviewManager.AGENT_NAME: InterviewManager().agent_callback,
            InterviewQuestionEvaluator.AGENT_NAME: InterviewQuestionEvaluator().agent_callback,
            InterviewQuestionGenerator.AGENT_NAME: InterviewQuestionGenerator().agent_callback,
            self.TOOLS_AGENT_NAME: ToolNode(tools=InterviewAgent.get_tools())
        }
        self.logger = init_logger()

    def _create_graph_and_add_nodes(self) -> StateGraph:
        """
        Create StateGraph instance with all nodes

        Return:
            new instance of StateGraph
        """
        self.logger.info("Creating nodes and builder ....")
        # //////////////// First Initialization ////////////////

        graph_builder = StateGraph(GraphState)

        # //////////////// Create Nodes ////////////////

        for agent_name, callback_func in self.nodes_dict.items():
            graph_builder.add_node(agent_name, callback_func)

        return graph_builder

    def _add_edges_with_conditions(self, graph_builder: StateGraph):
        """
        Add edges with conditions into StateGraph

        Args:
            graph_builder - StateGraph where add the edges
        """
        self.logger.info("Creating edges ....")
        # start with interview agent
        graph_builder.add_edge(START, InterviewManager.AGENT_NAME)

        # routing from manager. You can see that only manager can end the super step
        graph_builder.add_conditional_edges(InterviewManager.AGENT_NAME, InterviewManager.agent_router, {
            InterviewQuestionGenerator.AGENT_NAME: InterviewQuestionGenerator.AGENT_NAME,
            InterviewQuestionEvaluator.AGENT_NAME: InterviewQuestionEvaluator.AGENT_NAME,
            "END": END
        })
        # question generator and question evaluator return answer back to manager
        graph_builder.add_edge(InterviewQuestionGenerator.AGENT_NAME, InterviewManager.AGENT_NAME)
        graph_builder.add_edge(InterviewQuestionEvaluator.AGENT_NAME, InterviewManager.AGENT_NAME)

        # return tools to origin agent
        graph_builder.add_conditional_edges("tools", lambda state: state.last_agent)

    def create_super_step(self, user_message):
        graph_builder = self._create_graph_and_add_nodes()
        self._add_edges_with_conditions(graph_builder)
        graph_config = {
            "configurable": {
                "thread_id": uuid.uuid4()
            }
        }
        # add memory for whole session
        compiled_state_graph = graph_builder.compile(checkpointer=MemorySaver())
        interview_step_state = GraphState(
            messages=[{"role": "user", "content": user_message}]
        )
        compiled_state_graph.get_graph().draw_mermaid_png()
        result = compiled_state_graph.invoke(
            interview_step_state,
            config=graph_config
        )

        self.logger.info(f"Answer from app: {result['interview_manager_message']}")
        return result



if __name__ == '__main__':
    InterviewOrchestration().create_super_step("Hi")

    # # //////////////// Create Edges ////////////////
    #
    # # conditionally run tools if needed
    # graph_builder.add_conditional_edges("simple_node", tools_condition, "tools")
    # # this will help to loop around
    # graph_builder.add_edge("tools", "simple_node")
    # graph_builder.add_edge(START, "simple_node")
    #
    # # //////////////// Compile the Graph ////////////////
    #
    # # Compile the graph
    # graph_config = {}
    # if not compiled_state_graph:
    #     if memory_id:
    #         compiled_state_graph = graph_builder.compile(checkpointer=MemorySaver())
    #     else:
    #         compiled_state_graph = graph_builder.compile()
    #
    # if memory_id:
    #     graph_config = {
    #         "configurable": {
    #             "thread_id": memory_id
    #         }
    #     }
    # # //////////////// Create memory if memory id is setup ////////////////
    #
    # # //////////////// Invoke the Graph ////////////////
    # result = compiled_state_graph.invoke(
    #     interview_agent,
    #     config=graph_config
    # )
    # return result, compiled_state_graph
# graph = graph_builder.compile(checkpointer=sql_memory)
# display(Image(graph.get_graph().draw_mermaid_png()))
