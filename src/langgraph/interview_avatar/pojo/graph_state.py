from typing import Annotated, Optional, List, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel

from src.langgraph.interview_avatar.custom_types.manager_question_type import ManagerMessagePurpose
from src.langgraph.interview_avatar.custom_types.question_types import QuestionTypes


class GraphState(BaseModel):
    """
    Main class representing state within LangGraph
    """
    messages: Annotated[List[Any], add_messages]
    generated_question: Annotated[Optional[str], "Generated question"] = None
    # instructions which are send the question generator agent to create new question
    interview_manager_message: Annotated[Optional[str], "Instructions from interview manager"] = None
    # last agent which was process
    last_agent: Annotated[Optional[str], "Name of last agent which was process"] = None
    next_agent: Annotated[Optional[str], "Name of last next agent which need to be call"] = None
    # query from candidate. Especially some additional question to interview question
    candidate_query: Annotated[Optional[str], "Query from the user"] = None

    def get_last_candidate_message(self) -> str:
        """
        Get last message from candidate
        """

        # reorder from last index and find last humman message
        for message in self.messages[::-1]:
            if isinstance(message, HumanMessage):
                return message.content
        return ""


    def __str__(self):
        return str({
            "generated_question": self.generated_question,
            "interview_manager_message": self.interview_manager_message,
            "candidate_query": self.candidate_query,
            "last_agent": self.last_agent
        })

