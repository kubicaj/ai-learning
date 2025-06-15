from agents import Agent
from pydantic import BaseModel

from src.openai.common.pdf_utils import get_pdf_content


class QuestionFormat(BaseModel):
    question: str
    """This is the generated question"""
    complexity_level: int
    """This is level of complexity which was used for generating the question"""
    option_a: str
    """First possible answer"""
    option_b: str
    """Second possible answer"""
    option_c: str
    """Third possible answer"""
    option_d: str
    """Fourth possible answer"""


class QuestionGenerator(Agent):

    @staticmethod
    def _build_instructions():
        cv_content = get_pdf_content("resources/AWS Certified AI Practitioner Course AIF -C01.pdf")
        aws_course_name = "AWS Certified AI Practitioner Course AIF -C01"

        return f"""
            # Your task
            You are generate the exam question for {aws_course_name} with 4 possible answers/options but only one is correct. 
            The question will be provided to user which the user should answer on it.
            # How to generate the questions
            - Check the history of questions generate questions with the lowest similarities with historical questions, so you will give the user wide range of questions.
            - Generate question with 5 level complexity: 1 - trivial question, 5 - tricky question. 
            - User should read and answer it at time less then 150 seconds - depends on complexity. More complex question means the longer and more tricky question.
            - Generate questions randomly, but generate question with following ratio:
            1. level of complexity = 10% of questions
            2. level of complexity = 10% of questions
            3. level of complexity = 20% of questions
            4. level of complexity = 25% of questions
            5. level of complexity = 35% of questions
            # Context materials you can use for generating of the questions:
            {cv_content}
            """

    @staticmethod
    def build_agent() -> "QuestionGenerator":
        return QuestionGenerator(
            name="Question generator agent",
            instructions=QuestionGenerator._build_instructions(),
            model="gpt-4o-mini",
            output_type=QuestionFormat
        )

    @staticmethod
    def build_agent_as_tool():
        return QuestionGenerator.build_agent().as_tool(
            tool_name="agent_question_generator",
            tool_description="Agent for generate the exam question"
        )
