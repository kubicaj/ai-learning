from agents import Runner, trace
from dotenv import load_dotenv
import gradio as gr

from src.agents.aws_exams_learn_assistant.ai_agents.aws_exam_manager import AWSExamManager

load_dotenv(override=True)


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
        result = await Runner.run(AWSExamManager.build_agent(), message_prompt)
    return result.final_output


gr.ChatInterface(chat, type="messages").launch()
