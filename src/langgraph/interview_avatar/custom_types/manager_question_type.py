from enum import Enum


class ManagerMessagePurpose(Enum):
    """
    Enum of purpose of manager message
    """
    GENERATE_TECHNICAL_QUESTION = "generate_technical_question_by_asking_the_technical_lead"
    EVALUATE_TECHNICAL_QUESTION = "evaluate_technical_question_by_asking_the_technical_lead"
    SEND_CANDIDATE_QUESTION_TO_TECH_LEAD = "send_candidate_question_to_tech_lead"
    SEND_QUESTION_THE_CANDIDATE = "send_question_the_candidate"
    ANSWER_CANDIDATE_QUESTION = "answer_candidate_question"