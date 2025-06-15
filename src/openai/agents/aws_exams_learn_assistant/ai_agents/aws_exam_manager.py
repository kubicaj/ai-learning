from agents import Agent

from src.openai.agents.aws_exams_learn_assistant.ai_agents.question_evaluator import QuestionEvaluator
from src.openai.agents.aws_exams_learn_assistant.ai_agents.question_generator import QuestionGenerator
from src.openai.agents.aws_exams_learn_assistant.tools import save_answers_into_user_profile


class AWSExamManager(Agent):

    @staticmethod
    def _build_instructions():

        aws_course_name = "AWS Certified AI Practitioner Course AIF -C01"

        return f"""
        # Who you are
        1. You are AI manager in user chat and you are generating questions for AWS course {aws_course_name}.
        # What should you do:
        1. Analyze the chat history and check the last user input
        2. generating questions - Use agent_question_generator for this purpose. Send the agent also history of question, so agent will not repeat the same question again and again.
        3. Wait for user answer
        4. Always evaluate the answer if it is correct or not. Use question_evaluator_agent for this purpose
        5. Ask user, if he want next question. If yes then generate new question and repeat the all points
        6. At the end and only at the end when user confirm that do not want to next questions, 
        evaluate all the answers in the following way:
        - Print if user pass or not. User pass only in case he answered correct to more then 70 percent
        - Score - how much percentage was ok and how much was wrong
        - Feedback - what I should focus on
        6. Save the answer into user profile. Pass into arguments whole questions together with all options
    
    
        # History of questions
        Will be provided by your manager. Do not repeat the similar questions:
        """

    @staticmethod
    def build_agent() -> "AWSExamManager":
        return AWSExamManager(
            name="AWS exam manager",
            instructions=AWSExamManager._build_instructions(),
            tools=[
                QuestionGenerator.build_agent_as_tool(),
                QuestionEvaluator.build_agent_as_tool(),
                save_answers_into_user_profile
            ],
            model="gpt-4o-mini"
        )
