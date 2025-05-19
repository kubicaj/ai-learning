import uuid

import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import logging
import sys


class MyPersonalAvatarApp:
    """
    Class represent main Application class
    """

    def __init__(self):
        self.logger = self.init_logger()
        self.client = self.get_open_ai_client()
        self.cv_content = self.get_pdf_content("resources/CV_Juraj_Kubica.pdf")

    @staticmethod
    def init_logger():
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s][%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            stream=sys.stdout
        )
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        return logging.getLogger()

    @staticmethod
    def get_pdf_content(pdf_path: str) -> str:
        """
        Read PDF

        Args:
            - pdf_path: Path to PDF

        Returns:
            - text content of PDF
        """
        reader = PdfReader(pdf_path)
        pdf_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_text += text
        return pdf_text

    def get_open_ai_client(self) -> OpenAI:
        load_dotenv(override=True)
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if openai_api_key:
            self.logger.info(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
        else:
            self.logger.info("OpenAI API Key not set - please head to the troubleshooting guide in the setup folder")

        return OpenAI()

    @staticmethod
    def get_summary() -> str:
        return (
            "My name is Juraj Kubica. I am software engineer, with a lot of experience with Data and system integration"
            "I like to learn new thinks about AI")

    @staticmethod
    def get_personality() -> str:
        return (
            "I'm an ambitious person, but I don't cross my moral boundaries. "
            "In case of conflicts, I try to explain to people reasonably what it is about and give proven arguments. I don't try to bring feelings into conflicts, which helps me distance myself."
            "I don't communicate on topics I don't know about. I'm trying to listen more. I'm very introverted here."
            "But when I communicate with someone close to me, I'm more of an extrovert")

    def get_system_prompt(self) -> str:
        name = "Juraj Kubica"
        system_prompt = f"""
        ## What you should do
        
        You are acting as {name}. You are answering questions on {name}'s website, 
        particularly questions related to {name}'s career, background, skills and experience. 
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
        You are given a summary of {name}'s background and CV which you can use to answer questions. 
        
        ## Rules how to behave 
        
        1. Be professional and engaging, as if talking to a potential client or future employer who came across the website. 
        2. If you don't know the answer, say so. 
        3. If someone will ask you how are you, then answer in style that you have good day, because 
        4. user has contacted you and you have opportunity to answer the questions for him. 
        5. Be polite and introduce yourself at the start of conversation 
        6. In case of asking about salary or money exception please provide polite answer that it is sensitive topic to discuss here. 
        But provide contact to me so we can discuss about it personally 
        
        ## Summary:
        
        {self.get_summary()}
        
        ## CV:
        
        {self.cv_content}
        
        ## Your ultimate goal
        
        With this context, please chat with the user, always staying in character as {name}.
        Try to adopt your communication to {name} personality which is the following: {self.get_personality()}
        """
        return system_prompt

    def start_chat(self):

        def start_session():
            return str(uuid.uuid4())

        # define chat function
        def chat(message, history, top_p: float, temperature: float, session_id: str):
            """
            Main chat function
            """
            self.logger.info(f"[{session_id}] New message: {message}")
            messages = [{"role": "system", "content": self.get_system_prompt()}] + history + [
                {"role": "user", "content": message}]
            response = self.client.chat.completions.create(model="gpt-4o-mini", top_p=top_p, temperature=temperature,
                                                           messages=messages)
            answer = response.choices[0].message.content
            self.logger.info(f"[{session_id}] Answer: {answer}")
            return answer

        gr.ChatInterface(
            chat,
            type="messages",
            additional_inputs=[
                gr.Slider(0.0, 1.0, label="top_p", value=0.3,
                          info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.3)"),
                gr.Slider(0.0, 2.0, label="temperature", value=0.5,
                          info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.5)"),
                gr.State(start_session())
            ],
            title="Welcome 👋. I am Juraj Kubica's avatar. Ask me anything about my professional life.",
            theme=gr.themes.Ocean(),
            submit_btn="⬅ Send"
        ).launch()


if __name__ == '__main__':
    MyPersonalAvatarApp().start_chat()
