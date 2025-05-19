import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader


class MyPersonalAvatarApp:
    """
    Class represent main Application class
    """

    def __init__(self):
        self.client = self.get_open_ai_client()
        self.cv_content = self.get_pdf_content("resources/CV_Juraj_Kubica.pdf")

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

    @staticmethod
    def get_open_ai_client() -> OpenAI:
        load_dotenv(override=True)
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if openai_api_key:
            print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
        else:
            print("OpenAI API Key not set - please head to the troubleshooting guide in the setup folder")

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
        system_prompt = (f"You are acting as {name}. You are answering questions on {name}'s website, \
        particularly questions related to {name}'s career, background, skills and experience. \
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. \
        You are given a summary of {name}'s background and CV which you can use to answer questions. \
        Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
        If you don't know the answer, say so."
                         f"Be polite and introduce yourself at the start of conversation")
        system_prompt += f"\n\n## Summary:\n{self.get_summary()}\n\n## CV:\n{self.cv_content}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {name}."
        system_prompt += (f"Try to adopt your communication to {name} personality which is the following: "
                          f"{self.get_personality()}")
        return system_prompt

    def start_chat(self):
        # define chat function
        def chat(message, history, top_p: float, temperature: float):
            """
            Main chat function
            """
            messages = [{"role": "system", "content": self.get_system_prompt()}] + history + [
                {"role": "user", "content": message}]
            response = self.client.chat.completions.create(model="gpt-4o-mini", top_p=top_p, temperature=temperature,
                                                           messages=messages)
            return response.choices[0].message.content

        gr.ChatInterface(
            chat,
            type="messages",
            additional_inputs=[
                gr.Slider(0.0, 1.0, label="top_p", value=0.3,
                          info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.3)"),
                gr.Slider(0.0, 2.0, label="temperature", value=0.5,
                          info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.5)")
            ],
            title="Welcome 👋. I am Juraj Kubica's avatar. Ask me anything about my professional life.",
            theme=gr.themes.Ocean(),
            submit_btn="⬅ Send"
        ).launch()


if __name__ == '__main__':
    MyPersonalAvatarApp().start_chat()
