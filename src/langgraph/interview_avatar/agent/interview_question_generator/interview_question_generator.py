from typing import List

from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, HumanMessage

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
        # find last user message. Loop from the end because there are last queries from user
        user_query = None
        for message in graph_state.messages[::-1]:
            if isinstance(message, HumanMessage):
                user_query = message.content
                break
        # if there is nno user query then generate the question
        if not user_query:
            system_prompt = (
                f"""
                # Your role
                You are the Technical consultant which help during interview 
                
                # Your task
                - Generate  only one interview question 
                
                ## Rules, how to generate the question
                - Question has to be relevant to seniority of the role, which you can find in section `Position description`
                - Generate only and only one question
                - generate question which is type of {graph_state.generate_type_of_question.value}
                
                # Areas to interview
                - SQL
                - pyspark
                - databricks
                - AWS
                - Data engineering
                - Data architecture
                
                # Output structure
                
                The output of your answer HAVE TO BE formated as following:
                
                # Position description
                
                {POSITION_DESCRIPTION}
                
                ```text
                  
                ## Answer to user question
                
                - Add some additional notes here in case the user has some query. Do not fill this section in case the user has no questions
                - In case user has no additional queries leave this section blank
                
                ## Interview Question
                
                ### Question
                
                <Your generated question>
                
                ### Possible answers
                
                <Can be one or more acceptable answer for your generated question>
                ```
                
                """
            )
        else:
            system_prompt = """
            # Your role
            You are the Technical consultant which help during interview 
            
            # Your task
            - Answer user to query. Take into account your previously generated question 
            - Do not generate new question. Only answer the user question
            """
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
        generated_question = ""
        if isinstance(response, AIMessage) and not response.tool_calls:
            generated_question = response.content
        new_state = GraphState(
            messages=[response],
            generate_type_of_question=graph_state.generate_type_of_question,
            generated_question=generated_question
        )
        # process response
        return new_state


if __name__ == '__main__':
    compiled_state_graph = None
    for i in range(0, 4):
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
            generate_type_of_question=QuestionTypes.LIVE_CODING_QUESTION
        )
        result, compiled_state_graph = InterviewQuestionGenerator().call_as_standalone(
            graph_state,
            compiled_state_graph=compiled_state_graph)
        print(compiled_state_graph)
        print(result['generated_question'])
