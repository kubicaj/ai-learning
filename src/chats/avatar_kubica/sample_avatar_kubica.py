from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

from src.common.open_api_utils import get_open_ai_client
from src.common.pdf_utils import get_pdf_content

client = get_open_ai_client()
cv_content = get_pdf_content("resources/CV_Juraj_Kubica.pdf")

kubica_summary = ("My name is Juraj Kubica. I am software engineer, with a lot of experience with Data and system integration"
                  "I like to learn new thinks about AI")

name = "Juraj Kubica"
system_prompt = f"You are acting as {name}. You are answering questions on {name}'s website, \
particularly questions related to {name}'s career, background, skills and experience. \
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
You are given a summary of {name}'s background and CV which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer, say so."

system_prompt += f"\n\n## Summary:\n{kubica_summary}\n\n## CV:\n{cv_content}\n\n"
system_prompt += f"With this context, please chat with the user, always staying in character as {name}."

def chat(message, history):
    """
    Main chat function
    """
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(chat, type="messages").launch()
