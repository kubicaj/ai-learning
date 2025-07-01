from typing import List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, BaseMessage

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.types.question_types import QuestionTypes
from src.langgraph.llm.llm_factory import LLMFactory


def _load_position_description() -> str:
    """
    Load position description

    Return:
        str representation of position desc
    """
    with open('resources/position_description.md', 'r') as file:
        return file.read()


POSITION_DESCRIPTION = _load_position_description()


class InterviewQuestionGenerator(InterviewAgent):
    """
    Agent responsible for generating of the interview questions for particular topic.
    Agent will also provide the best answer for the question
    """

    def _create_interview_generator_prompt(self, graph_state: GraphState) -> List[BaseMessage]:
        """
        Create interview generator prompt

        Args:
            graph_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        self.logger.debug("Create interview generator prompt...")
        system_prompt = (
            f"""
            # Your role
            You are the HR professional which is able to generate the interview questions
            
            # Your task
            - Generate the only one interview question and also the relevant and description answer for generating question for the position
            describe bellow in section `Position description`
            - Setup the questions based on seniority of the role, which you can find in section `Position description`
            - Answer only and only by one question and answer on your generated question.
            - Check the history and first, start with soft questions then continue with more technical. 
            
            # Type of question to generate
            Now generate question which is type of {graph_state.generate_type_of_question}
            
            # Output structure
            The output of your answer HAVE TO BE formated as following
            ```text
            ## Question
            <Your generated question>
            
            ## Answer
            <Can be one or more acceptable answer for your generated question>
            ```
            
            # Position description
            
            {POSITION_DESCRIPTION}
            """
        )
        return graph_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback(self, graph_state: GraphState) -> GraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        self.logger.debug(f"Invoking agent {self.__class__.__name__}")
        # create and invoke LLM agent
        open_ai_llm = LLMFactory.get_chat_open_ai_llm()
        llm_with_tools = open_ai_llm.bind_tools(self.get_tools())
        response = llm_with_tools.invoke(input=self._create_interview_generator_prompt(graph_state))

        new_state = GraphState(messages=[response], generate_type_of_question=graph_state.generate_type_of_question)
        # process response
        return new_state


if __name__ == '__main__':
    initial_state = GraphState(messages=[{"role": "user", "content": "Start"}],
                               generate_type_of_question=QuestionTypes.LIVE_CODING_QUESTION)
    result = InterviewQuestionGenerator().call_as_standalone(initial_state)
    print(result)
