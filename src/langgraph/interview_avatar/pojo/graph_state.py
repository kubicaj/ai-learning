from typing import Annotated, Optional, List, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from src.langgraph.interview_avatar.custom_types.manager_question_type import ManagerQuestionType
from src.langgraph.interview_avatar.custom_types.question_types import QuestionTypes


class GraphState(BaseModel):
    """
    Main class representing state within LangGraph
    """
    messages: Annotated[List[Any], add_messages]
    generate_type_of_question: Annotated[Optional[QuestionTypes], "Type of question which needs to be generated"] = None
    generated_question: Annotated[Optional[str], "Generated question"] = None
    # instructions which are send the question generator agent to create new question
    interview_manager_message: Annotated[Optional[str], "Instructions from interview manager"] = None
    # manager question type
    manager_question_type: Annotated[Optional[ManagerQuestionType], "Type of question from manager"] = None
    # last agent which was process
    last_agent: Annotated[Optional[str], "Name of last agent which was process"] = None

    def get_last_candidate_message(self) -> str:
        """
        Get last message from candidate
        """

        candidate_message_order: int = 0
        for message in self.messages[::-1]:
            candidate_message_order += 1
            if isinstance(message, HumanMessage) and candidate_message_order != 1:
                return message.content
        return ""

