import gradio as gr
import functools

from src.langgraph.interview_avatar.interview_config import InterviewConfig
from src.langgraph.interview_avatar.interview_app import InterviewApp

POSITIONS = list(InterviewConfig.get_active_instance().all_open_positions.values())

PRIMARY_COLOR = "#224488"
BG = "#f4f4fb"
MAX_COLUMNS = 4  # Max cards per row


def load_position_detail(idx):
    """
    Show detail modal with content from the selected position.

    Args:
        idx (int): Index of selected position

    Returns:
        tuple: (Show modal, content, title)
    """
    open_position = POSITIONS[idx]
    return gr.update(visible=True), open_position.open_position_content, open_position.position_title


def hide_detail():
    """Hide and clear the detail modal."""
    return gr.update(visible=False), "", ""


def set_interview_buttons(disable=True):
    """
    Enable/disable all interview start buttons.

    Args:
        disable (bool): True disables buttons, False enables.

    Returns:
        list: gradio update for each button
    """
    return [gr.update(interactive=(not disable)) for _ in POSITIONS]


def start_interview(idx):
    """
    Prepare interface for a new interview.

    Args:
        idx (int): Position index

    Returns:
        tuple: Set interview area visible, set position name, index, disable buttons, clear chat and state.
    """
    initial_message = (
        f"Hi. You are here because you apply for position {POSITIONS[idx].position_title}. Can we start please?")
    interview_app_app = setup_interview_app(POSITIONS[idx].position_identifier)
    return (
        gr.update(visible=True),  # Show interview chat area
        POSITIONS[idx].position_title,  # Show interview title
        idx,  # Internal state index for position
        *set_interview_buttons(disable=True),  # Disable all "Start interview" buttons
        [{"role": "assistant", "content": initial_message}],  # Clear chat display
        [{"role": "assistant", "content": initial_message}],  # Clear chat state/history
        interview_app_app  # create new interview application
    )


def show_confirm_modal():
    """
    Show confirmation dialog for closing interview.
    """
    return gr.update(visible=True)


def hide_confirm_modal():
    """
    Hide confirmation modal.
    """
    return gr.update(visible=False)


def end_interview(interview_app: InterviewApp, history: list[dict]):
    """
    Return UI state to 'no interview running'. Enable buttons etc.
    """

    return (
        gr.update(visible=False),  # interview_chat hidden
        "",  # position name cleared
        -1,  # index cleared
        *set_interview_buttons(disable=False),  # enable all buttons
        gr.update(visible=False),  # confirmation modal hidden
    )


def setup_interview_app(position_to_interview):
    """
    Setup interview APP each time the new interview is started

    Args:
        position_to_interview (str): Identifier of position to inteview

    Returns:
        (InterviewApp) new interview application instance
    """
    interview_app = InterviewApp(position_to_interview)
    interview_app.create_graph()
    return interview_app


def chat_fn(user_input, history: list[dict], interview_app: InterviewApp) -> tuple[list[dict], list[dict]]:
    """
    Simple handler: HR stub reply to every user turn.

    Args:
        user_input (str): The user's message.
        history (list): The running list of chat turns.
        interview_app (str): interview app

    Returns:
        tuple: Updated visible chat log, updated internal state.
    """
    result = interview_app.invoke_user_query(user_input, history)
    history = history + [{"role": "user", "content": user_input}] + [result]
    return history, history


def chunk(seq, size):
    """
    Split `seq` into sublists of length `size`. Used to render grid of cards.

    Args:
        seq (list): List to chunk.
        size (int): Max chunk size.

    Yields:
        list: Next chunk.
    """
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


# ========== UI Layout ==========
with gr.Blocks(css=f"""
.centered-row {{
    display: flex;
    justify-content: center !important;
    flex-wrap: wrap;
}}
.card {{
    background: {BG};
    border-radius: 16px;
    box-shadow: 0 2px 10px #dde2ef;
    border: 2px solid #dde5fc;
    margin: 10px;
    min-width: 220px;
    max-width: 340px;
    flex: 1 1 22%;
    padding: 26px 20px 20px 20px;
    text-align: center;
    box-sizing: border-box;
}}
.card-title {{ color: {PRIMARY_COLOR}; font-size: 1.22em; margin-bottom: 8px; font-weight: 700;}}
.card-desc {{ color: #444; font-size: 1em; }}
""") as demo:
    # ===== Header =====
    gr.Markdown(
        "## Career Portal\n"
        "_Below are currently open positions:_"
    )

    # ===== Create job cards in a grid below the header =====
    show_btns = []
    interview_btns = []
    for chunk_positions in chunk(POSITIONS, MAX_COLUMNS):
        with gr.Row(elem_classes="centered-row"):
            for pos in chunk_positions:
                with gr.Group(elem_classes="card"):
                    gr.Markdown(f"<div class='card-title'>{pos.position_title}</div>")
                    gr.Markdown(f"<div class='card-desc'>{pos.position_short_summary}</div>")
                    btn_det = gr.Button("Show detail")
                    btn_int = gr.Button("Start interview", variant="primary")
                    show_btns.append(btn_det)
                    interview_btns.append(btn_int)

    # ===== Modal: Position Detail =====
    with gr.Group(visible=False) as detail_group:
        detail_title = gr.Markdown("**Position Detail**", elem_id="modal-title")
        detail_md = gr.Markdown("", elem_id="modal-md")
        btn_close = gr.Button("Close detail", variant="stop")
    btn_close.click(hide_detail, outputs=[detail_group, detail_md, detail_title])

    # ===== Modal: End-Interview Confirm =====
    with gr.Group(visible=False) as confirm_modal:
        confirm_msg = gr.Markdown("Are you sure you want to end the interview?")
        btn_yes = gr.Button("Yes, end interview", variant="stop")
        btn_no = gr.Button("No, continue")

    # ======== Persistent app state (hidden by default) ========
    interview_chat = gr.Column(visible=False)  # Contains entire chat panel, hidden until interview started
    chosen_position_name = gr.State("")  # Keeps selected name for chat
    interview_application = gr.State(None)
    chosen_position_idx = gr.State(-1)  # Index of running interview

    # ======== Interview Chat UI (rendered AFTER cards/grid) ========
    # This block (and its contents) are always below the cards!
    with interview_chat:  # Only appears when set visible!
        pos_label = gr.Markdown("", elem_id="interview-title")  # Chat section title (dynamic)
        chatbot = gr.Chatbot(type="messages")  # Chat message area (history is managed via code)
        state = gr.State([])  # List of chat turns (messages)
        msg = gr.Textbox(label="Your response / question...")  # User input
        send_btn = gr.Button("Send", variant="primary")  # Chat send button
        btn_end = gr.Button("End interview", variant="stop")  # End-interview button

        # When user sends, run chat_fn which returns (chatbot_content, history)
        send_btn.click(chat_fn, [msg, state, interview_application], [chatbot, state])


        # Update the chat section title according to position on change
        def update_title(pos_name):
            return f"### Interview chat for **{pos_name}** position"


        chosen_position_name.change(update_title, chosen_position_name, pos_label)

    # ====== Setup event connections (outside of panels!) ======
    # Position detail modal
    for idx, btn in enumerate(show_btns):
        btn.click(functools.partial(load_position_detail, idx), outputs=[detail_group, detail_md, detail_title])

    # Interview start: resets chat & disables all interview buttons
    for idx, btn in enumerate(interview_btns):
        btn.click(
            functools.partial(start_interview, idx),
            outputs=[interview_chat, chosen_position_name, chosen_position_idx] + interview_btns + [chatbot, state,
                                                                                                    interview_application]
        )

    # End interview confirmation and finish buttons
    btn_end.click(show_confirm_modal, outputs=[confirm_modal])
    btn_no.click(hide_confirm_modal, outputs=[confirm_modal])
    btn_yes.click(
        end_interview,
        inputs=[interview_application, state],
        outputs=[interview_chat, chosen_position_name, chosen_position_idx] + interview_btns + [confirm_modal]
    )

# ========= RUN UI =========
demo.launch()
