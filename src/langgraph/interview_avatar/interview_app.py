import gradio as gr

from src.langgraph.interview_avatar.interview_orchestration import InterviewOrchestration


def setup_interview_app():
    print("Setuping the interview application")
    interview_orchestration = InterviewOrchestration("position_description")
    interview_orchestration.create_graph()
    return interview_orchestration


async def process_candidate_message(user_input, history, interview_orchestration: InterviewOrchestration):
    result = interview_orchestration.invoke_user_query(user_input, history)
    history = history + [{"role": "user", "content": user_input}] + [result]
    return history, history


with gr.Blocks(title="Interview agent", theme=gr.themes.Default(primary_hue="emerald")) as ui:
    gr.Markdown("## Welcome to interview")

    # init new application
    interview_orchestration = gr.State()
    ui.load(setup_interview_app, [], [interview_orchestration])

    chatbot = gr.Chatbot(
        label="Interview manager",
        height=300,
        type="messages"
    )
    message_from_candidate = gr.Textbox(label="Your message")
    history = gr.State([])

    with gr.Row():
        send_message_button = gr.Button("Send message", variant="primary")

    send_message_button.click(
        fn=process_candidate_message,
        inputs=[message_from_candidate, history, interview_orchestration],
        outputs=[chatbot, history]
    )

    message_from_candidate.submit(
        fn=process_candidate_message,
        inputs=[message_from_candidate, history, interview_orchestration],
        outputs=[chatbot, history]
    )

ui.launch()
