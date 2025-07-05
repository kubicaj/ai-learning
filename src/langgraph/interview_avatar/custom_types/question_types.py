from enum import Enum


class QuestionTypes(Enum):
    """
    Enum represents type of interview questions
    """

    SOFT_SKILL_QUESTION = "soft skill question"
    TECHNICAL_QUESTION = "technical question"
    LIVE_CODING_QUESTION = "live coding question"
