from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from src.common.open_api_utils import get_open_ai_client
from src.common.pdf_utils import get_pdf_content

client = get_open_ai_client()
cv_content = get_pdf_content("../avatar_kubica/resources/fastapi_tutorial.pdf")

name = "FastAPI avatar"
system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to python FastAPI framework problematic. \
Your responsibility is to help your team mates with their issues with Fast API. \
You are given a documentation of FastAPI which is your knowledge base to use to answer questions. \
Be professional and engaging, as if talking to your collage which is asking you about help."

system_prompt += f"\n\n## documentation of FastAPI:\n{cv_content}\n\n"
system_prompt += f"With this context, please chat with the team mates, always staying in character as {name}."

def chat(message, history):
    """
    Main chat function
    """
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch()
