from typing import List

from langchain_core.messages import SystemMessage, BaseMessage

from src.langgraph.interview_avatar.agent.interview_agent import InterviewAgent
from src.langgraph.interview_avatar.pojo.interview_graph_state import InterviewGraphState
from src.langgraph.llm.llm_factory import LLMFactory


class InterviewAdministrator(InterviewAgent):
    """
    Agent to evaluate the whole interview
    """

    AGENT_NAME = "interview_administrator"

    def _create_system_prompt(self, interview_state: InterviewGraphState) -> List[BaseMessage]:
        """
        Create interview evaluator prompt

        Args:
            interview_state - actual state of graph

        Args:
            SystemMessage with prompt
        """
        # if there is nno user query then generate the question
        system_prompt = self.agent_prompt_templates["agent_prompt"].format(**{
            "position_description": self._interview_config.get_position_content(self.chosen_position),
            "candidate_cv": self._interview_config.candidate_cv
        })
        return interview_state.messages + [SystemMessage(content=system_prompt)]

    def agent_callback_implementation(self, interview_state: InterviewGraphState) -> InterviewGraphState:
        """
        Agent callback method. More info see InterviewAgent.agent_callback
        """
        open_ai_llm = LLMFactory.get_chat_open_ai_llm(self.AGENT_NAME)
        messages = self._create_system_prompt(interview_state)
        response = open_ai_llm.invoke(input=messages)

        # get all communication and message from interview and send it to company
        self._send_result_interview(interview_state, response.content)
        # ADD next agent as END
        new_state = interview_state.create_copy(
            {"role": "assistant",
             "content": "Ok, I created the summary and send it to company. "
                        "Please inform candidate that interview ended "},
            last_agent=self.AGENT_NAME,
        )
        return new_state


    def _send_result_interview(self, interview_state: InterviewGraphState, administrator_output: str):
        """
        Send results about interview

        Args:
            interview_state (InterviewGraphState) - graph state
            administrator_output (str) - administrator output with summary
        """
        all_message_history = \
            [f"[{item.message_time}] <{item.subject_role}> : {item.message_content} \n" for one_iter in
             interview_state.iteration for item in one_iter]

        final_output = (f"## Messages history\n\n {all_message_history} \n\n ## Summary of interview \n\n  "
                        f"{administrator_output}")
        # TODO - send email
        self.logger.info(final_output)
