from typing import Annotated, Optional, List, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel


def append_historical_iterations(previous_iteration_messages: list | None, new_iteration_messages: list) -> list:
    """
    TODO
    """

    if len(new_iteration_messages) == 1 and len(new_iteration_messages[0]) == 1:
        # starting new superstep
        # Take the history and append new iteration
        previous_iteration_messages.append(new_iteration_messages[0])
        new_iteration_messages = previous_iteration_messages

    return new_iteration_messages


class HumanToAiIteration(BaseModel):
    """
    Class represent one iteration human -> <agents> -> human
    """
    message_time: Annotated[str, "Current date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message_content: Annotated[str, "Message content"] = ""
    subject_role: Annotated[str, "Name of agent or human"] = ""


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
    # messages accumulation per one human to agent iteration with specific format
    iteration: Annotated[list[list[HumanToAiIteration]], append_historical_iterations] = None
    # session id
    session_id: Annotated[str, "id of session"]

    def create_copy(self, new_message: dict, last_agent: str = None, next_agent: str = None,
                    generated_question: str = None, interview_manager_message: str = None):
        """
        Create new copy of state

        Args:
            new_message (dict): new state message
            last_agent (str): name of last agent
            next_agent (str): name of next agent
            generated_question (str): new generated question
            interview_manager_message (str): new message from manager

        Return:
            (InterviewGraphState) copy of current object instance
        """
        if not self.iteration:
            self.iteration = [[]]

        if isinstance(new_message, dict):
            self.iteration[-1].append(HumanToAiIteration(
                message_content=new_message.get("content"),
                subject_role=self.last_agent
            ))
        if isinstance(new_message, BaseMessage):
            self.iteration[-1].append(HumanToAiIteration(
                message_content=new_message.content,
                subject_role=self.last_agent
            ))

        return InterviewGraphState(
            messages=[new_message],
            generated_question=self.generated_question if generated_question is None else generated_question,
            interview_manager_message=self.interview_manager_message if interview_manager_message is None else
            interview_manager_message,
            last_agent=self.last_agent if last_agent is None else last_agent,
            next_agent=self.next_agent if next_agent is None else next_agent,
            candidate_query=self.candidate_query,
            iteration=self.iteration,
            session_id=self.session_id
        )

    def get_last_candidate_message(self) -> str:
        """
        Get last message from candidate

        Return:
            (str) - str message representation
        """
        # reorder from last index and find last human message
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
