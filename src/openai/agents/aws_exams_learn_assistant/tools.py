"""
Module with function tools
"""
from agents import function_tool


@function_tool
def save_answers_into_user_profile(list_of_questions_and_answers: list[str]):
    """
    Save questions and user answer into user profile.
    As an input there is list of all questions and answers
    """
    print(list_of_questions_and_answers)
    return {
        "result": "success"
    }
