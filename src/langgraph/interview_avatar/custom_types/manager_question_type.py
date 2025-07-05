from enum import Enum


class ManagerQuestionType(Enum):
    """
    Enum of questions manager can ask
    """
    ASK_TECHNICAL_LEAD_TO_GET_INTERVIEW_QUESTION = "ask_technical_lead_to_get_the_question"
    ASK_TECHNICAL_LEAD_TO_EVALUATE_QUESTION = "ask_technical_lead_to_evaluate_the_question"
    ASK_USER_QUESTION = "ask_user_question"
    SEND_USER_GENERATED_QUESTION = "send_user_generated_question"