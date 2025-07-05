from typing import List

from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.config_loader import POSITION_DESCRIPTION, TOPICS_TO_INTERVIEW
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.custom_types.question_types import QuestionTypes
from src.langgraph.llm.llm_factory import LLMFactory


class InterviewQuestionGenerator(InterviewAgent):
    """
    Agent responsible for generating of the interview questions for particular topic.
    Agent will also provide the best answer for the question
    """

    AGENT_NAME = "interview_question_generator"

    @staticmethod
    def get_prompt_names() -> list[str]:
        """
        Overwrite default prompt names
        """
        # default is agent prompt
        return ["question_generator_prompt", "user_query_prompt"]

    def _create_interview_generator_prompt(self, interview_state: GraphState, user_query: str) -> List[BaseMessage]:
        """
        Create interview generator prompt

        Args:
            graph_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        self.logger.debug("Create interview generator prompt...")
        # if there is nno user query then generate the question
        if not user_query:
            system_prompt = self.agent_prompt_templates["question_generator_prompt"].format(
                **{
                    "generate_type_of_question": interview_state.generate_type_of_question.value,
                    "position_description": POSITION_DESCRIPTION,
                    "interview_manager_instructions": interview_state.interview_manager_message,
                    "topics_to_interview": TOPICS_TO_INTERVIEW
                }
            )
        else:
            system_prompt = self.agent_prompt_templates["user_query_prompt"]
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback(self, interview_state: GraphState) -> GraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        self.logger.info(f"Invoking agent {self.__class__.__name__}")
        # create and invoke LLM agent
        open_ai_llm = LLMFactory.get_chat_open_ai_llm()
        llm_with_tools = open_ai_llm.bind_tools(self.get_tools())

        # find last user message. Loop from the end because there are last queries from user
        user_query = interview_state.get_last_candidate_message()

        response = llm_with_tools.invoke(input=self._create_interview_generator_prompt(interview_state, user_query))

        # save the previous generated question
        generated_question = interview_state.generated_question
        if isinstance(response, AIMessage) and not response.tool_calls:
            generated_question = response.content

        new_state = GraphState(
            messages=[response],
            generate_type_of_question=interview_state.generate_type_of_question,
            generated_question=generated_question,
            interview_manager_message=interview_state.interview_manager_message,
            last_agent=self.AGENT_NAME
        )
        # process response
        return new_state


if __name__ == '__main__':
    compiled_state_graph = None
    for i in range(0, 1):
        print(f"======================= ITERATION {i} ======================= \n")
        user_contents = [
            "",
            "Can I use also different programing language?",
            "Can you provide me the expected output?",
            "But give me some sample of output please?"
        ]
        print(f"User query: {user_contents[i]} \n")
        graph_state = GraphState(
            messages=[{"role": "user", "content": user_contents[i]}],
            generate_type_of_question=QuestionTypes.TECHNICAL_QUESTION,
            interview_manager_message="Give me some simple question about Databricks"
        )
        result, compiled_state_graph = InterviewQuestionGenerator().call_as_standalone(
            graph_state,
            compiled_state_graph=compiled_state_graph,
            memory_id="1"
        )
        print(compiled_state_graph)
        print(f"QUESTION {result['generated_question']}")
        print(f"USER ANSWER: {result['user_query_answer']}")
