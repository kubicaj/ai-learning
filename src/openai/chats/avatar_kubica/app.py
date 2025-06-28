import uuid

import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader

from src.openai.chats.avatar_kubica.guardrails import Guardrails, ValidationError
from src.openai.common.logger import init_logger


class MyPersonalAvatarApp:
    """
    Class represent main Application class
    """

    LLM_MODEL_TYPE = "gpt-4o-mini"

    def __init__(self):
        self.logger = init_logger()
        self.client = self.get_open_ai_client()
        self.cv_content = self.get_pdf_content("resources/CV_Juraj_Kubica.pdf")
        self._guardrails = Guardrails(self.client, self.LLM_MODEL_TYPE, 500)

    def get_open_ai_client(self) -> OpenAI:
        """
        Create new OPEN AI client

        Return:
            Open AI object instance
        """
        load_dotenv(override=True)
        openai_api_key = os.getenv('OPENAI_API_KEY')

        if openai_api_key:
            self.logger.info(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
        else:
            self.logger.info("OpenAI API Key not set - please head to the troubleshooting guide in the setup folder")

        return OpenAI()

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
    def get_avatar_role() -> str:
        """
        Create role of avatar

        Return:
            str representation of avatar role
        """
        return (
            "My name is Juraj Kubica. I am software engineer, with a lot of experience with Data and system integration"
            "I like to learn new thinks about AI")

    @staticmethod
    def get_personality() -> str:
        """
        Describe user personality

        Return:
            str representation user personality
        """
        return (
            "I'm an ambitious person, but I don't cross my moral boundaries. "
            "In case of conflicts, I try to explain to people reasonably what it is about and give proven arguments. I don't try to bring feelings into conflicts, which helps me distance myself."
            "I don't communicate on topics I don't know about. I'm trying to listen more. I'm very introverted here."
            "But when I communicate with someone close to me, I'm more of an extrovert")

    def get_system_prompt(self) -> str:
        """
        Create main system prompt

        Return:
            system prompt
        """
        name = "Juraj Kubica"
        system_prompt = f"""
        ## What you should do
        
        You are acting as {name}. You are answering questions on {name}'s website, 
        particularly questions related to {name}'s career, background, skills and experience. 
        Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
        You are given a summary of {name}'s background and CV which you can use to answer questions. 
        
        ## My preferred roles
        I prefer the following roles:
        - Senior Data engineer
        - Data Architect
        - Enterprise Architect
        - Technical Lead
        - AI engineer (I am very eager to learn all new about AI and I am training and learning daily about this area)
        - AWS cloud engineer
        - AWS cloud Architect
        
        But in general I am open to whatever IT technical role which will involve me into new technologies enhance my 
        knowledge and mainly to the role which will bring value to customer  
        
        ## Rules how to behave 
        
        1. Be professional and engaging, as if talking to a potential client or future employer who came across the website. 
        2. If you don't know the answer, say so. 
        3. If someone will ask you how are you, then answer in style that you have good day, because 
        4. user has contacted you and you have opportunity to answer the questions for him. 
        5. Be polite and introduce yourself at the start of conversation 
        6. In case of asking about salary or money exception please provide polite answer that it is sensitive topic to discuss here. 
        But provide contact to me so we can discuss about it personally 
        
        ## Summary:
        
        {self.get_avatar_role()}
        
        ## CV:
        
        {self.cv_content}
        
        ## Your ultimate goal
        
        With this context, please chat with the user, always staying in character as {name}.
        Try to adopt your communication to {name} personality which is the following: {self.get_personality()}
        """
        return system_prompt

    def start_chat(self):
        """
        Main chat function
        """

        def generate_session_id() -> str:
            """
            Method create new uuid which represent session ID
            """
            return str(uuid.uuid4())

        def chat(message, history, top_p: float, temperature: float, session_id: str) -> str:
            """
            Main chat message which is call each time user send input

            Args:
                message - user input
                history - history of chat
                top_p - top P parameter for LLM
                temperature - temperature parameter for LLM
                session_id - id of session
            """

            self.logger.info(f"[{session_id}] New message: {message}")
            # reduce history first. Limit is set up to 30 so it means 15 questions and 15 answers
            history = self._guardrails.reduce_history(history, max_size_history=30)
            # validate message
            try:
                self._guardrails.validate(message)
            except ValidationError as validation_error:
                self.logger.error(f"[{session_id}] Validation error: {str(validation_error)}")
                return str(validation_error)

            # continue if message is valid
            messages = [{"role": "system", "content": self.get_system_prompt()}] + history + [
                {"role": "user", "content": message}]
            response = self.client.chat.completions.create(model=self.LLM_MODEL_TYPE, top_p=top_p,
                                                           temperature=temperature,
                                                           messages=messages)
            answer = response.choices[0].message.content
            self.logger.info(f"[{session_id}] Answer: {answer}")
            return answer

        # Contact info HTML (customize as needed)
        CONTACTS = """
        <h3>Contact Juraj Kubica</h3>
        <ul style="font-size:16px;">
          <li>📧 <b>Email:</b> <a href="mailto:kubica.juro@gmail.com">kubica.juro@gmail.com</a></li>
          <li>🌐 <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/juraj-kubica-897a3590" target="_blank">https://www.linkedin.com/in/juraj-kubica-897a3590</a></li>
          <li>📱 <b>Phone:</b> +420 601 121 964</li>
        </ul>
        """

        with gr.Blocks(theme=gr.themes.Ocean()) as cv_app:
            gr.Markdown(
                "# 👨‍💻 Welcome! I'm Juraj Kubica's Avatar\n\nAsk me anything about my professional journey below. If you'd like, download my CV or contact me directly!")
            with gr.Row():
                gr.HTML(CONTACTS)
                gr.File(value="resources/CV_Juraj_Kubica.pdf", label="Download CV 📄", interactive=True)
            gr.Markdown("---")

            gr.ChatInterface(
                chat,
                type="messages",
                show_progress = "full",
                additional_inputs=[
                    gr.Slider(
                        0.0, 1.0, label="top_p", value=0.3, render=False,
                        info="Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.3)"),
                    gr.Slider(
                        0.0, 2.0, label="temperature", value=0.5, render=False,
                        info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.5)"),
                    gr.State(generate_session_id())
                ],
                title=None,
                submit_btn="⬅ Send"
            )

        cv_app.launch()


if __name__ == '__main__':
    MyPersonalAvatarApp().start_chat()
