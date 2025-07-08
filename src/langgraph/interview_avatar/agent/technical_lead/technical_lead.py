from typing import List
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.config_loader import POSITION_DESCRIPTION, TOPICS_TO_INTERVIEW
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.custom_types.question_types import QuestionTypes
from src.langgraph.llm.llm_factory import LLMFactory


class TechnicalLead(InterviewAgent):
    """
    Agent responsible for evaluating the user answer
    """

    AGENT_NAME = "technical_lead"

    def _create_system_prompt(self, interview_state: GraphState) -> List[BaseMessage]:
        """
        Create interview evaluator prompt

        Args:
            interview_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        generated_question = (
            interview_state.generated_question) if interview_state.generated_question else "No question here"

        self.logger.debug("Create interview evaluator prompt...")
        system_prompt = self.agent_prompt_templates["agent_prompt"].format(
            **{
                "generated_question": generated_question,
                "candidate_question": interview_state.candidate_query,
                "position_description": POSITION_DESCRIPTION,
                "topics_to_interview": TOPICS_TO_INTERVIEW,
                "interview_manager_message": interview_state.interview_manager_message,
                "answer_to_question": interview_state.candidate_query,
                "generate_or_not_possible_answers": "DO NOT",
                "additional_note_about_task": ""
            })
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback_implementation(self, interview_state: GraphState) -> GraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        # create and invoke LLM agent
        open_ai_llm = LLMFactory.get_chat_open_ai_llm()
        llm_with_tools = open_ai_llm.bind_tools(self.get_tools())
        response = llm_with_tools.invoke(input=self._create_system_prompt(interview_state))
        generated_question = ""
        if isinstance(response, AIMessage) and not response.tool_calls:
            generated_question = response.content
        new_state = GraphState(
            messages=[response],
            generated_question=generated_question,
            last_agent=self.AGENT_NAME
        )
        # process response
        return new_state

