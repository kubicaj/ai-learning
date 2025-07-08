"""
Modul containing tool for Google search
"""
from dotenv import load_dotenv
from langchain_core.tools import Tool
from langchain_community.utilities import GoogleSerperAPIWrapper

load_dotenv(override=True)

serper_api_wrapper = GoogleSerperAPIWrapper()

tool_search_for_interview = Tool(
    name="search",
    func=serper_api_wrapper.run,
    description="Use for searching some inspiration for interview questions or checking the right answers"
)
