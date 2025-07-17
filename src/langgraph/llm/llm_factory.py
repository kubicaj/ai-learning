from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# get to load environment variables
load_dotenv(override=True)

_chat_open_ai_model: dict[str, ChatOpenAI] = {}


class LLMFactory:
    """
    Factory to get LLM clients of various purpose
    """

    @staticmethod
    def get_chat_open_ai_llm(llm_id = "llm_id", llm_model_type="gpt-4o-mini") -> ChatOpenAI:
        """
        Create new OpenAI client

        Prerequisites:
            Define OPENAI_API_KEY key in .env file in your repo

        Return:
            ChatOpenAI instance
        """
        # you can use regular OpenAI SDK but because we are in LangGraph ecosystem then it is better to show how to call
        # OpenAI llm using LangChain libraries gpt-4.1
        global _chat_open_ai_model
        if not _chat_open_ai_model.get(llm_id):
            _chat_open_ai_model[llm_id] = ChatOpenAI(model=llm_model_type)
        return _chat_open_ai_model[llm_id]
