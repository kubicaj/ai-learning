from enum import Enum
from typing import List, Annotated

from langchain_core.messages import SystemMessage, BaseMessage
from pydantic import BaseModel

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent, POSITION_DESCRIPTION, CANDIDATE_CV
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.types.interview_stage import InterviewStage
from src.langgraph.interview_avatar.types.question_types import QuestionTypes
from src.langgraph.llm.llm_factory import LLMFactory
from typing import Optional, Literal


class TypeOfManagerQuestion(Enum):
    """
    Enum of questions manager can ask
    """
    ASK_CV_QUESTION = "ask_cv_question"
    ASK_TECHNICAL_LEAD_TO_QUESTION = "ask_technical_lead_to_the_question"
    ASK_TECHNICAL_LEAD_TO_EVALUATE_QUESTION = "ask_technical_lead_to_evaluate_the_question"
    ASK_USER_QUESTION = "ask_user_question"


class ManagerOutput(BaseModel):
    """
    Structured answer of manager agent
    """
    type_of_question: Annotated[TypeOfManagerQuestion, "Type of question manager is asking"]
    stage: Annotated[InterviewStage, "Stage of interview where manager is focus"]
    manager_input: Annotated[Optional[str], "Question of manager to user or to technical lead"]


class InterviewManager(InterviewAgent):
    """
    Agent to manage the interview
    """

    def _create_system_prompt(self, interview_state: GraphState) -> List[BaseMessage]:
        """
        Create interview manager prompt

        Args:
            interview_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        self.logger.debug("Create interview evaluator prompt...")
        # if there is nno user query then generate the question
        system_prompt = self.agent_prompt_templates["agent_prompt"].format(**{
            "position_description": POSITION_DESCRIPTION,
            "candidate_cv": CANDIDATE_CV
        })
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback(self, interview_state: GraphState) -> GraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        self.logger.debug(f"Invoking agent {self.__class__.__name__}")
        # create and invoke LLM agent
        open_ai_llm = LLMFactory.get_chat_open_ai_llm()
        llm_with_structured_output = open_ai_llm.with_structured_output(ManagerOutput)
        response = llm_with_structured_output.invoke(input=self._create_system_prompt(interview_state))

        new_state = GraphState(
            messages=[
                {"role": "assistant", "content": f"Question from interview manager: {response.manager_input}"}],
            generate_type_of_question=interview_state.generate_type_of_question,
            generated_question=interview_state.generated_question
        )
        # process response
        return new_state


if __name__ == '__main__':
    compiled_state_graph = None
    for i in range(0, 5):
        print(f"======================= ITERATION {i} ======================= \n")
        user_contents = [
            "Hi",
            "Ok we can continue",
            "Perfect continue",
            "I like data engineering",
            "I do not know"
        ]
        print(f"User query: {user_contents[i]} \n")
        graph_state = GraphState(
            messages=[{"role": "user", "content": user_contents[i]}],
            generate_type_of_question=QuestionTypes.TECHNICAL_QUESTION,
            generated_question="",
            interview_manager_instructions="Give me some simple question about Databricks"
        )
        result, compiled_state_graph = InterviewManager().call_as_standalone(
            graph_state,
            compiled_state_graph=compiled_state_graph,
            memory_id="1"
        )
        print(compiled_state_graph)
