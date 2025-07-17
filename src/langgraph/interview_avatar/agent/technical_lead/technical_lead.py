from typing import List
from langchain_core.messages import SystemMessage, BaseMessage, AIMessage, ToolMessage

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.pojo.interview_graph_state import InterviewGraphState
from src.langgraph.llm.llm_factory import LLMFactory


class TechnicalLead(InterviewAgent):
    """
    Agent responsible for evaluating the user answer
    """

    AGENT_NAME = "technical_lead"

    def __init__(self, additional_note_about_task: str = "", number_of_generated_questions: int = 1):
        """
        Args:
            number_of_generated_questions - Number of generated questions per one shot
            additional_note_about_task - additional note to the task
        """
        super().__init__()
        self.number_of_generated_questions = number_of_generated_questions
        self.additional_note_about_task = additional_note_about_task

    def _create_system_prompt(self, interview_state: InterviewGraphState) -> List[BaseMessage]:
        """
        Create interview evaluator prompt

        Args:
            interview_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        generated_question = (
            interview_state.generated_question) if interview_state.generated_question else "No question here"

        self.logger.debug("Create interview evaluator prompt...")
        system_prompt = self.agent_prompt_templates["agent_prompt"].format(
            **{
                "generated_question": generated_question,
                "candidate_question": interview_state.candidate_query,
                "position_description": self._interview_config.get_position_content(self.chosen_position),
                "interview_manager_message": interview_state.interview_manager_message,
                "answer_to_question": interview_state.candidate_query,
                "generate_or_not_possible_answers": "DO NOT",
                "additional_note_about_task": self.additional_note_about_task,
                "number_of_generated_questions": self.number_of_generated_questions
            })
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback_implementation(self, interview_state: InterviewGraphState) -> InterviewGraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        # create and invoke LLM agent
        open_ai_llm = LLMFactory.get_chat_open_ai_llm()
        llm_with_tools = open_ai_llm.bind_tools(self.get_tools())
        messages = self._create_system_prompt(interview_state)
        response = llm_with_tools.invoke(input=messages)
        generated_question = ""
        if isinstance(response, AIMessage) and not response.tool_calls:
            generated_question = response.content

        if isinstance(response, AIMessage) and response.tool_calls:
            final_response = self.process_tool_response(
                response, messages, llm_with_tools
            )
            return InterviewGraphState(
                messages=[final_response],
                generated_question=final_response.content,
                agent_iterations=interview_state.agent_iterations,
                last_agent=self.AGENT_NAME
            )

        # return in case of calling LLM without tools
        return InterviewGraphState(
            messages=[response],
            generated_question=generated_question,
            agent_iterations=interview_state.agent_iterations,
            last_agent=self.AGENT_NAME
        )
