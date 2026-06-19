from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv(override=True)


def ensure_vectorstore():
    """Build the vector store on first startup if it is not present."""
    vector_db = Path(__file__).parent / "vector_db"
    if (vector_db / "chroma.sqlite3").exists():
        return

    from implementation.ingest import build_vectorstore, create_chunks, fetch_documents

    docs = fetch_documents()
    chunks = create_chunks(docs)
    build_vectorstore(chunks)


ensure_vectorstore()

from implementation.answer import answer_question


def format_context(docs):
    """Format retrieved context for display in the side panel."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        parts.append(f"Source: {source}\n\n{doc.page_content}\n\n{'-' * 60}\n")
    return "\n".join(parts)


def chat(message, history):
    """Handle a user message and return the updated UI state."""
    history = history or []

    answer, docs = answer_question(message, history)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history, format_context(docs)


def main():
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.blue,
    )

    custom_css = """
    #app-title h1 {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #059669 0%, #2563eb 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    }
    #app-subtitle p {
        text-align: center;
        font-size: 1.05rem;
        opacity: 0.85;
    }
    """

    with gr.Blocks(title="Iziko Cloud Solutions - Expert Assistant") as ui:
        gr.Markdown(
            "# Iziko Cloud Solutions - Expert Assistant",
            elem_id="app-title",
        )
        gr.Markdown(
            "Ask me anything about Iziko Cloud Solutions: products, contracts, "
            "employees, culture, or careers.",
            elem_id="app-subtitle",
        )

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=500)
                with gr.Row():
                    message = gr.Textbox(
                        placeholder="e.g. What does IzikoGuard do?",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Column():
                context_box = gr.Textbox(label="Retrieved Context", lines=25)

        submit_args = {
            "fn": chat,
            "inputs": [message, chatbot],
            "outputs": [message, chatbot, context_box],
        }
        message.submit(**submit_args)
        send_btn.click(**submit_args)

    ui.launch(inbrowser=False, server_name="0.0.0.0", theme=theme, css=custom_css)


if __name__ == "__main__":
    main()
