from typing import Annotated

from openai import OpenAI
from pydantic import BaseModel


class ValidationError(Exception):
    """
    Raise in case of Validation error of input message
    """
    pass


class ExpressiveTermEvalOutput(BaseModel):
    """
    Structure output of System Content Controller
    """
    is_message_appropriate: Annotated[bool, "Flag indicates if the message is appropriate (=True) or not (=False)"]
    answer_explanation: Annotated[str, "Explanation why the message is or is not appropriate"]


class Guardrails:
    """
    Custom Guardrails for CV APP
    """

    def __init__(self, open_ai_client: OpenAI, llm_model_type: str, max_message_length: int = 300):
        self._open_ai_client = open_ai_client
        self._llm_model_type = llm_model_type
        self._max_message_length = max_message_length

    @staticmethod
    def reduce_history(history: list[dict], max_size_history: int = 25) -> list[dict]:
        """
        In order to save cost and prevet overlapping of contex, do not allow the history to be bigger than
        max_size_history arg

        Args:
            history - history list
            user_input_message - max size of history

        Return:
            Reduced history
        """
        if history and len(history) > max_size_history:
            return history[max_size_history:]
        return history

    def validate(self, user_input_message: str):
        """
        Main validator message
        Args:
            user_input_message - user input message

        Return:
            None

        Raise:
            ValidationError: in case of validation error
        """
        self._validate_max_length(user_input_message)
        self._validate_expressive_terms(user_input_message)

    def _validate_max_length(self, user_input_message: str):
        """
        Check if input_message is not bigger than max_message_length. If yes raise ValidationError

        Args:
            user_input_message - user input message

        Return:
            None

        Raise:
            ValidationError: in case of validation error
        """
        if len(user_input_message) > self._max_message_length:
            raise ValidationError(f"Message is bigger then {self._max_message_length}. Please reduce user input")

    def _validate_expressive_terms(self, user_input_message: str):
        """
        Validate if user input contains some expressive terms

        Args:
            user_input_message - user input message

        Return:
            None

        Raise:
            ValidationError: in case of validation error
        """
        guardrails_instruction = f"""
        # Role
        System Content Controller
        
        # Objective
        Check that the input message does not contain obscene, expressive, 
        offensive or otherwise inappropriate language.
        """
        messages = [
            {
                "role": "system",
                "content": guardrails_instruction
            },
            {
                "role": "user",
                "content": user_input_message
            },
        ]

        response = self._open_ai_client.beta.chat.completions.parse(
            model=self._llm_model_type,
            messages=messages,
            response_format=ExpressiveTermEvalOutput
        )
        if not response.choices[0].message.parsed.is_message_appropriate:
            raise ValidationError(f"{response.choices[0].message.parsed.answer_explanation}")
