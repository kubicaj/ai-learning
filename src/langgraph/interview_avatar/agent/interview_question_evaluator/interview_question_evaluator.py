from typing import List
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.pojo.graph_state import GraphState
from src.langgraph.interview_avatar.custom_types.question_types import QuestionTypes
from src.langgraph.llm.llm_factory import LLMFactory


class InterviewQuestionEvaluator(InterviewAgent):
    """
    Agent responsible for evaluating the user answer
    """

    AGENT_NAME = "interview_question_evaluator"

    def _create_system_prompt(self, interview_state: GraphState) -> List[BaseMessage]:
        """
        Create interview evaluator prompt

        Args:
            interview_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        self.logger.debug("Create interview evaluator prompt...")
        system_prompt = self.agent_prompt_templates["agent_prompt"].format(
            **{
                "generated_question": interview_state.generated_question,
                "user_message": interview_state.get_last_candidate_message(),
            })
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback(self, interview_state: GraphState) -> GraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        self.logger.info(f"Invoking agent {self.__class__.__name__}")
        # create and invoke LLM agent
        open_ai_llm = LLMFactory.get_chat_open_ai_llm()
        llm_with_tools = open_ai_llm.bind_tools(self.get_tools())
        response = llm_with_tools.invoke(input=self._create_system_prompt(interview_state))
        generated_question = ""
        if isinstance(response, AIMessage) and not response.tool_calls:
            generated_question = response.content
        new_state = GraphState(
            messages=[response],
            generate_type_of_question=interview_state.generate_type_of_question,
            generated_question=generated_question,
            last_agent=self.AGENT_NAME
        )
        # process response
        return new_state


if __name__ == '__main__':
    compiled_state_graph = None
    for i in range(0, 1):
        print(f"======================= ITERATION {i} ======================= \n")
        user_contents = [
            """Yes, it is my daily work. First I am trying to reach them personally because I believe that personal communication is the best. 
            If it is not possible because of location or whatever other reason, I am trying to find some free space in their calendar and schedule the meeting.
            I am sending the ideas and thoughts in advance so stakeholders have time to prepare.
            During the meeting I am trying to present my results and try to find any trades off if needed. It is crucial because more people have more ideas and opinions and there is minimal chance that we will agree on all points
            After meeting I am summarizing meeting notes where I am pointing on crucial alignment we achieve and trying to arange another meetings/call if there are still any open topics
            Output of this alligment should not be results only, but also the next steps and task for all participants. Expect sending meeting notes to email I am also putting the intermediate (or final) results into confluence page
            where everyone has access and comment it"""
        ]
        print(f"User query: {user_contents[i]} \n")
        graph_state = GraphState(
            messages=[{"role": "user", "content": user_contents[i]}],
            generate_type_of_question=QuestionTypes.LIVE_CODING_QUESTION,
            generated_question="""# Interview Question
            
            ## Question
            Can you describe a time when you had to collaborate with multiple stakeholders to deliver a data engineering project? How did you ensure effective communication and alignment on goals?
            
            ## Possible answers
            An ideal answer could include:
            - A specific project example, detailing the stakeholders involved (data scientists, product managers, etc.).
            - The strategies used for communication (regular meetings, documentation, feedback loops).
            - How the candidate managed differing priorities and ensured everyone was aligned on the project goals.
            - The outcome of the project and any lessons learned from the collaboration process.
            
            ## Question note
            - This question aims to assess the candidate's soft skills in collaboration, communication, and stakeholder management, which are essential for a Senior Data Engineer. The interviewer should look for clear examples of past experiences, demonstrating the candidate's ability to navigate complex interpersonal dynamics and drive projects to completion successfully.
            """
        )
        result, compiled_state_graph = InterviewQuestionEvaluator().call_as_standalone(
            graph_state,
            compiled_state_graph=compiled_state_graph,
            memory_id="1"
        )
        print(compiled_state_graph)
        print(result['generated_question'])
