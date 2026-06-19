from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document

load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
llm = ChatOpenAI(temperature=0, model_name=MODEL)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing Iziko Cloud Solutions,
a South African cloud and IT services company headquartered in Cape Town.

Use ONLY the retrieved context below to answer questions about Iziko's products,
contracts, employees, and company information. If the context does not contain
the answer, say so honestly rather than guessing.

Context:
{context}
"""


def resolve_query(query: str, history: list) -> str:
    """Combine the latest user turn with the new query so retrieval has more signal."""
    last_user = None
    if history:
        for turn in reversed(history):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
    if last_user:
        return f"{last_user}\n{query}"
    return query


def fetch_context(query: str) -> list[Document]:
    """Retrieve the top-k context documents for a query."""
    return retriever.invoke(query)


def answer_question(query: str, history=None):
    """
    Generate an answer using retrieved context plus a lightweight chat history.

    History is a list of {"role": "user"|"assistant", "content": "..."}.
    Returns (answer_text, retrieved_documents).
    """
    resolved_query = resolve_query(query, history or [])
    docs = fetch_context(resolved_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]

    if history:
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=query))

    response = llm.invoke(messages)
    return response.content, docs
