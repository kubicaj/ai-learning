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

    def _create_interview_generator_prompt(self, graph_state: GraphState, user_query: str) -> List[BaseMessage]:
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
                - You have to follow the instructions from interview manager. The instructions from interview manager has priority
                
                # Additional instruction from interview manager
                
                {graph_state.interview_manager_instructions}
                
                # Areas to interview
                - SQL
                - pyspark
                - databricks
                - AWS
                - Data engineering
                - Data architecture
                
            
                # Position description
                
                {POSITION_DESCRIPTION}
                
                # Output structure
                
                The output of your answer HAVE TO BE formated as following:
                
                ```text
                # Interview Question
                
                ## Question
                <Your generated question>
                
                ## Possible answers
                <Can be one or more acceptable answer for your generated question>
                
                ## Question note
                - Add some additional notes here. For example what is expected output, what do you expect user will do etc
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

        # find last user message. Loop from the end because there are last queries from user
        user_query = None
        for message in graph_state.messages[::-1]:
            if isinstance(message, HumanMessage):
                user_query = message.content
                break

        response = llm_with_tools.invoke(input=self._create_interview_generator_prompt(graph_state, user_query))

        # save the previous generated question
        generated_question = graph_state.generated_question
        user_query_answer = ""
        if isinstance(response, AIMessage) and not response.tool_calls:
            if user_query:
                user_query_answer = response.content
            else:
                generated_question = response.content

        new_state = GraphState(
            messages=[response],
            generate_type_of_question=graph_state.generate_type_of_question,
            generated_question=generated_question,
            user_query_answer=user_query_answer,
            interview_manager_instructions=graph_state.interview_manager_instructions
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
            interview_manager_instructions="Give me some simple question about Databricks"
        )
        result, compiled_state_graph = InterviewQuestionGenerator().call_as_standalone(
            graph_state,
            compiled_state_graph=compiled_state_graph,
            memory_id="1"
        )
        print(compiled_state_graph)
        print(f"QUESTION {result['generated_question']}")
        print(f"USER ANSWER: {result['user_query_answer']}")
