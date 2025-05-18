from typing import Any

from src.chats.aws_exams_learn_assistant.tools.base_tool import AITool


class SaveAnswersIntoUserProfile(AITool):
    """
    Function to save answers into user profile
    """

    def get_function_name(self) -> str:
        return "save_answers_into_user_profile"

    def call_function(self, **args) -> Any:
        list_of_questions_and_answers = args["list_of_questions_and_answers"]
        print(list_of_questions_and_answers)
        pass

    def get_tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.get_function_name(),
                "description": "Save questions and user answer into user profile. As a input there is list of all questions and answers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "list_of_questions_and_answers": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["list_of_questions_and_answers"],
                    "additionalProperties": False
                }
            }
        }
