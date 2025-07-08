from typing import List, Annotated, Literal

from langchain_core.messages import SystemMessage, BaseMessage
from pydantic import BaseModel

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.agent.technical_lead.technical_lead import TechnicalLead
from src.langgraph.interview_avatar.config_loader import POSITION_DESCRIPTION, CANDIDATE_CV
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.custom_types.interview_stage import InterviewStage
from src.langgraph.llm.llm_factory import LLMFactory


class ManagerOutput(BaseModel):
    """
    Structured answer of manager agent
    """
    action_to_take: Annotated[
        Literal[
            "asking_technical_lead", "sending_message_to_candidate"], "What action the interview manager is performing"]
    stage: Annotated[InterviewStage, "Stage of interview where manager is focus"]
    manager_message: Annotated[str, "Manager message: to user or to technical lead"]


class InterviewManager(InterviewAgent):
    """
    Agent to manage the interview
    """

    AGENT_NAME = "interview_manager"

    def _create_system_prompt(self, interview_state: GraphState) -> List[BaseMessage]:
        """
        Create interview manager prompt

        Args:
            interview_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        answer_from_technical_lead = "No answer from technical lead. Feel free to ask"
        if interview_state.last_agent == TechnicalLead.AGENT_NAME and interview_state.generated_question:
            answer_from_technical_lead = interview_state.generated_question
        self.logger.debug("Create interview evaluator prompt...")
        # if there is nno user query then generate the question
        system_prompt = self.agent_prompt_templates["agent_prompt"].format(**{
            "position_description": POSITION_DESCRIPTION,
            "candidate_cv": CANDIDATE_CV,
            "answer_from_technical_lead": answer_from_technical_lead
        })
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    @staticmethod
    def agent_router(interview_state: GraphState) -> str:
        """
        Create router (where the route the agent answers) for langgraph

        Args:
            interview_state: current state

        Return:
            name of next agent
        """
        return interview_state.next_agent

    def agent_callback_implementation(self, interview_state: GraphState) -> GraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        open_ai_llm = LLMFactory.get_chat_open_ai_llm(self.AGENT_NAME)
        llm_with_structured_output = open_ai_llm.with_structured_output(ManagerOutput)
        response: ManagerOutput = llm_with_structured_output.invoke(input=self._create_system_prompt(interview_state))

        # if manager is asking about the question then send the instructions to tech lead
        new_state = GraphState(
            messages=[
                {"role": "assistant", "content": f"{response.manager_message}"}],
            generated_question=interview_state.generated_question,
            interview_manager_message=response.manager_message,
            candidate_query=None,
            last_agent=self.AGENT_NAME,
            next_agent=TechnicalLead.AGENT_NAME if response.action_to_take == "asking_technical_lead" else "END"
        )
        # process response
        return new_state
