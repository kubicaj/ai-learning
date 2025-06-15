import json
from abc import ABC, abstractmethod
from typing import Any


class AITool(ABC):

    @abstractmethod
    def get_tool_definition(self) -> dict:
        pass

    @abstractmethod
    def call_function(self, **args) -> Any:
        pass

    @abstractmethod
    def get_function_name(self) -> str:
        pass
