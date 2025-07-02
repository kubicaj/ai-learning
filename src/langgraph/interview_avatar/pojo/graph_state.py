from typing import Annotated, Optional, List, Any, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel

from src.langgraph.interview_avatar.types.question_types import QuestionTypes


class GraphState(BaseModel):
    """
    Main class representing state within LangGraph
    """
    messages: Annotated[List[Any], add_messages]
    generate_type_of_question: Annotated[QuestionTypes, "Type of question which needs to be generated"]
    generated_question: Annotated[str, "Generated question"] = None
