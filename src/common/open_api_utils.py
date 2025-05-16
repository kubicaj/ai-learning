import os
from openai import OpenAI
from dotenv import load_dotenv

def load_env():
    """
    Load environment first
    """
    load_dotenv(override=True)

def get_open_ai_client() -> OpenAI:
    load_env()
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set - please head to the troubleshooting guide in the setup folder")

    return OpenAI()