import json

from agents import Agent, Runner, trace, function_tool
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
from src.common.open_api_utils import get_open_ai_client
from src.common.pdf_utils import get_pdf_content

load_dotenv(override=True)

client = get_open_ai_client()
cv_content = get_pdf_content("resources/AWS Certified AI Practitioner Course AIF -C01.pdf")

name = "AWS exam manager"

aws_course_name = "AWS Certified AI Practitioner Course AIF -C01"

question_generator_instructions = f"""
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
"""
agent_question_generator = Agent(
    name="Question generator agent",
    instructions=question_generator_instructions,
    model="gpt-4o-mini"
)

question_evaluator_agent = """
As a input you are getting the question from exam manager and answer from User
Evaluate the answer if it is correct or not. 
Output has to follow the format:
<Evaluation in form correct/incorrect>
<Correct answer with explanation>
<For each incorrect option explain why it is incorrect. You can not miss that!>
"""
agent_answer_evaluator = Agent(
    name="Answer evaluator agent",
    instructions=question_evaluator_agent,
    model="gpt-4o-mini"
)

exam_manager_instructions = f"""
# Who you are
1. You are AI manager in user chat and you are generating questions for AWS course {aws_course_name}.
# What should you do:
1. Analyze the chat history and check the last user input
2. generating questions - Use agent_question_generator for this purpose. Send the agent also history of question, so agent will not repeat the same question again and again.
3. Wait for user answer
4. Evaluate the answer if it is correct or not. Use question_evaluator_agent for this purpose
5. Ask user, if he want next question. If yes then generate new question and repeat the all points
6. At the end and only at the end when user confirm that do not want to next questions, 
evaluate all the answers in the following way:
- Print if user pass or not. User pass only in case he answered correct to more then 70 percent
- Score - how much percentage was ok and how much was wrong
- Feedback - what I should focus on
6. Save the answer into user profile. Pass into arguments whole questions together with all options
"""

agent_question_generator_tool = agent_question_generator.as_tool(
    tool_name="agent_question_generator",
    tool_description="Agent for generate the exam question"
)
agent_answer_evaluator_tool = agent_answer_evaluator.as_tool(
    tool_name="agent_answer_evaluator",
    tool_description="Agent for evaluate the user answers for generated questions"
)


@function_tool
def save_answers_into_user_profile(list_of_questions_and_answers: list[str]):
    """
    Save questions and user answer into user profile. As a input there is list of all questions and answers
    """
    print(list_of_questions_and_answers)
    return {
        "result": "success"
    }


exam_manager = Agent(
    name="AWS exam manager",
    instructions=exam_manager_instructions,
    tools=[agent_question_generator_tool, agent_answer_evaluator_tool, save_answers_into_user_profile],
    model="gpt-4o-mini"
)


async def chat(message, history):
    """
    Main chat function
    """
    history_prompt = "There is no history yet. Generate the first question"
    if history:
        history_prompt = ""
        his_index = 0
        for his_message in history:
            role = ("User input:" if his_index % 2 == 0 else "AI manager:")
            history_prompt += f"{role} {his_message.get('content')} \n"
            his_index += 1
    message_prompt = f"""
    # History of chat
    {history_prompt}
    
    # Message from user
    {message}
    """
    with trace("AWS EXAM Certification learning"):
        result = await Runner.run(exam_manager, message_prompt)
    return result.final_output


gr.ChatInterface(chat, type="messages").launch()
