import json

from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from src.chats.aws_exams_learn_assistant.tools.base_tool import AITool
from src.chats.aws_exams_learn_assistant.tools.save_answers_into_user_profile import SaveAnswersIntoUserProfile
from src.common.open_api_utils import get_open_ai_client
from src.common.pdf_utils import get_pdf_content

client = get_open_ai_client()
cv_content = get_pdf_content("resources/AWS Certified AI Practitioner Course AIF -C01.pdf")

name = "AWS exam generator"
aws_course_name = "AWS Certified AI Practitioner Course AIF -C01"
system_prompt = \
    (f"# Who you are:"
     f"{name}. "
     f"# What should you do:"
     f"1. - You are generating questions for AWS course {aws_course_name}, You are generate only the question"
     f"with 4 questions where only one is correct. You are providing this input to user which is interacting with you."
     f"Generate questions with 5 level complexity. 1 - trivial question, 5 - tricky question. All the questions, user"
     f"should be read and answer to less then 90 seconds. Based on complexity"
     f""
     f"2. - Wait for user answer"
     f"3. - Evaluate the answer if it is correct or not. Output should be in format"
     f"<Evaluation in form correct/incorrect>"
     f"<Correct answer with explanation>"
     f"<For each incorrect option explain why it is incorrect> "
     f"4. Ask user, if he want next question. If yes then generate new question and repeat the all points. "
     f"If not, then evaluate the answers in the following way:"
     f"- Print if user pass or not. User pass only in case he answered correct to more then 70 percent"
     f"- Score - how much percentage was ok and how much was wrong"
     f"- Feedback - what I should focus on"
     f"5. Save the answer into user profile. Pass into arguments whole questions together with all options")


def get_tools() -> dict[str, AITool]:
    save_answers_into_user_profile = SaveAnswersIntoUserProfile()
    return {save_answers_into_user_profile.get_function_name(): save_answers_into_user_profile}


def chat(message, history):
    """
    Main chat function
    """
    messages = [{"role": "system", "content": system_prompt}] + \
               history + \
               [{"role": "user", "content": message}]
    tool_instances = get_tools()
    tools = [tool.get_tool_definition() for tool in tool_instances.values()]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

    finish_reason = response.choices[0].finish_reason
    # If the LLM wants to call a tool, we do that!
    if finish_reason == "tool_calls":
        message = response.choices[0].message
        tool_calls = message.tool_calls
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_instances[tool_name].call_function(**json.loads(tool_call.function.arguments))

    return response.choices[0].message.content


gr.ChatInterface(chat, type="messages").launch()
