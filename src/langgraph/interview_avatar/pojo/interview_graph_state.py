from typing import Annotated, Optional, List, Any

from langchain_core.messages import HumanMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


def increment_number_by_one(existing_state_number: int | None, new_state_number: int) -> int:
    """
    Reducer to increase number in state file

    Args:
        existing_state_number - state number from previous state
        new_state_number - state number from current state

    Return:
        (int) new number
    """
    return (existing_state_number or 0) + 1


class InterviewGraphState(BaseModel):
    """
    Main class representing state within LangGraph
    """
    # list of messages
    messages: Annotated[List[Any], add_messages]
    # generated question for candidate
    generated_question: Annotated[Optional[str], "Generated question"] = "No question"
    # instructions which are send the question generator agent to create new question
    interview_manager_message: Annotated[Optional[str], "Instructions from interview manager"] = None
    # last agent which was process
    last_agent: Annotated[Optional[str], "Name of last agent which was process"] = None
    # next agent to route
    next_agent: Annotated[Optional[str], "Name of last next agent which need to be call"] = None
    # query from candidate. Especially some additional question to interview question
    candidate_query: Annotated[Optional[str], "Query from the user"] = None
    # agent iterations
    agent_iterations: Annotated[int, increment_number_by_one] = 0

    def get_last_candidate_message(self) -> str:
        """
        Get last message from candidate

        Return:
            (str) - str message representation
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
