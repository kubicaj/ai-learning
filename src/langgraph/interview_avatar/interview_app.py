import gradio as gr

from src.langgraph.interview_avatar.interview_orchestration import InterviewOrchestration


def setup_interview_app():
    interview_orchestration = InterviewOrchestration()
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
    msg = gr.Textbox(label="Your message")
    history = gr.State([])

    send_btn = gr.Button("Send")

    send_btn.click(
        fn=process_candidate_message,
        inputs=[msg, history, interview_orchestration],
        outputs=[chatbot, history]
    )


    with gr.Row():
        send_message_button = gr.Button("Send message", variant="primary")

    # message_from_candidate.submit(process_candidate_message, [interview_orchestration, message_from_candidate, chatbot], [chatbot, message_from_candidate])
    # send_message_button.click(process_candidate_message, [interview_orchestration, message_from_candidate, chatbot], [chatbot, message_from_candidate])

ui.launch()



# def chat(user_input, history, is_admin=False):
#     if is_admin:
#         response = f"[ADMIN MODE] You said: '{user_input}'"
#     else:
#         response = f"You said: '{user_input}'"
#
#     history = history + [
#         {"role": "user", "content": user_input},
#         {"role": "assistant", "content": response}
#     ]
#     return history, history
#
# with gr.Blocks() as ui:
#     interview_orchestration = gr.State()
#     ui.load(setup_interview_app, [], [interview_orchestration])
#     chatbot = gr.Chatbot(type="messages")
#     msg = gr.Textbox(label="Your message")
#     is_admin = gr.Checkbox(label="Admin Mode")
#     state = gr.State([])
#
#     send_btn = gr.Button("Send")
#
#     send_btn.click(
#         fn=chat,
#         inputs=[msg, state, is_admin],
#         outputs=[chatbot, state]
#     )
# ui.launch()