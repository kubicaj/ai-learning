from agents import Agent
from pydantic import BaseModel


class OptionAnswerFormat(BaseModel):
    answer_option: str
    """Answer option - can be A, B, C or D"""
    answer_explanation: str
    """Explain why the answer is correct or incorrect"""


class AnswerFormat(BaseModel):
    correct_answer: OptionAnswerFormat
    """This is the generated question"""
    incorrect_answers: list[OptionAnswerFormat]
    """List of incorrect answer together with explanation"""


class QuestionEvaluator(Agent):

    @staticmethod
    def _build_instructions():
        return """
        As a input you are getting the question from exam manager and answer from User
        Evaluate the answer if it is correct or not. 
        Output has to follow the format:
        <Evaluation in form correct/incorrect>
        <Correct answer with explanation>
        <For each incorrect option explain why it is incorrect. You can not miss that!>
        """

    @staticmethod
    def build_agent() -> "QuestionEvaluator":
        return QuestionEvaluator(
            name="Answer evaluator agent",
            instructions=QuestionEvaluator._build_instructions(),
            model="gpt-4o-mini",
            output_type=AnswerFormat
        )

    @staticmethod
    def build_agent_as_tool():
        return QuestionEvaluator.build_agent().as_tool(
            tool_name="agent_answer_evaluator",
            tool_description="Agent for evaluate the user answers for generated questions"
        )
