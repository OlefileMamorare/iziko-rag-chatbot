# Iziko Cloud Solutions - RAG Project Planning Document

## 1. Project Objective and Use Case

### Objective

Build a Retrieval-Augmented Generation (RAG) chatbot that can answer natural-language questions about a fictional South African IT company, **Iziko Cloud Solutions**, by retrieving information from a curated internal knowledge base and grounding a Large Language Model's response in that retrieved context.

### Use Case

The chatbot is positioned as an **internal expert assistant** for Iziko employees, prospective customers, and new joiners. Typical questions include:

- "What does IzikoGuard do, and what does the Enterprise Tier cost?"
- "Who is the technical lead on IzikoConnect?"
- "What is the contract value of the Highveld Energy IzikoGuard contract?"
- "Which compliance certifications does Iziko hold?"

This use case is well suited to RAG because:

- The corpus is small but high-value (product specs, contracts, people, culture).
- Answers must be **grounded in source documents** and traceable.
- New documents (e.g. new contracts) are added regularly and must be available without retraining the LLM.

## 2. Dataset and Data Sources

The knowledge base is a synthetic but realistic corpus designed to mirror what a real South African IT services company would maintain on a corporate wiki. It lives entirely in [knowledge-base/](knowledge-base) and is organised by document type:

| Folder                                               | Documents | Description                                                          |
| ---------------------------------------------------- | --------- | -------------------------------------------------------------------- |
| [knowledge-base/company](knowledge-base/company)     | 4         | Company background, vision, culture, careers                         |
| [knowledge-base/products](knowledge-base/products)   | 4         | One file per product: IzikoConnect, IzikoGuard, IzikoData, IzikoFlow |
| [knowledge-base/employees](knowledge-base/employees) | 6         | Bios for key employees (CEO, Head of Engineering, CFO, etc.)         |
| [knowledge-base/contracts](knowledge-base/contracts) | 8         | Anonymised enterprise customer contracts                             |

All files are Markdown (`.md`) with UTF-8 encoding. They were authored specifically for this assignment to avoid any reuse of confidential or copyrighted material.

## 3. RAG Architecture and Workflow

### High-Level Architecture

```
                +---------------------+
User question -> | Gradio chat UI      | <- Retrieved context panel
                +----------+----------+
                           |
                           v
                +----------+----------+
                | answer_question()   |
                |  (implementation/   |
                |   answer.py)        |
                +----+-----------+----+
                     |           |
              top-k  |           | grounded prompt
             retrieval           |
                     v           v
       +-------------+----+   +--+------------------+
       | Chroma vector DB |   | OpenAI Chat model   |
       | (vector_db/)     |   | gpt-4.1-nano        |
       +-------+----------+   +---------------------+
               ^
               | embed + persist (one-off)
               |
       +-------+-------+
       | ingest.py     |
       | (knowledge-   |
       |  base/*.md)   |
       +---------------+
```

### Workflow

1. **Ingestion** ([implementation/ingest.py](implementation/ingest.py))
   - Load every `.md` file under [knowledge-base/](knowledge-base) using LangChain's `DirectoryLoader` + `TextLoader`.
   - Tag each document with a `doc_type` metadata field equal to its parent folder name.
   - Split documents into chunks of 500 characters with 200-character overlap using `RecursiveCharacterTextSplitter`.
   - Embed each chunk with **OpenAI `text-embedding-3-large`**.
   - Persist all chunks and their embeddings to a local **Chroma** vector store at `vector_db/`.

2. **Retrieval + Generation** ([implementation/answer.py](implementation/answer.py))
   - For each user query, optionally combine it with the previous user turn (`resolve_query`) to provide better retrieval signal.
   - Retrieve the top **k = 5** relevant chunks from Chroma.
   - Build a `SystemMessage` containing the company-specific system prompt and the retrieved context.
   - Append prior conversation turns (lightweight memory) plus the new user message.
   - Call **OpenAI `gpt-4.1-nano`** via `ChatOpenAI` with `temperature=0`.

3. **User Interface** ([app.py](app.py))
   - A Gradio `Blocks` UI with a chat panel on the left and a "Retrieved Context" inspector on the right.
   - Designed to run locally or on a Hugging Face Space (binds to `0.0.0.0`).

4. **Evaluation** ([evaluator.py](evaluator.py), [evaluation/eval.py](evaluation/eval.py))
   - 28 test questions in [evaluation/tests.jsonl](evaluation/tests.jsonl) covering direct facts and spanning questions.
   - **Retrieval metrics:** Mean Reciprocal Rank (MRR), Normalized DCG (nDCG@10), keyword coverage.
   - **Answer metrics:** LLM-as-a-judge scoring on accuracy, completeness, and relevance (1-5).

## 4. Tools, Frameworks, and Technologies

| Layer          | Technology                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| Language       | Python 3.11+                                                                                          |
| Orchestration  | LangChain (`langchain-community`, `langchain-openai`, `langchain-chroma`, `langchain-text-splitters`) |
| Vector store   | Chroma (local, persisted to disk)                                                                     |
| Embeddings     | OpenAI `text-embedding-3-large`                                                                       |
| LLM            | OpenAI `gpt-4.1-nano` (via `ChatOpenAI` and `litellm` for the judge)                                  |
| UI             | Gradio 4.x                                                                                            |
| Evaluation     | Custom MRR/nDCG + LLM-as-a-judge via LiteLLM + Pydantic                                               |
| Deployment     | Hugging Face Spaces (Gradio SDK)                                                                      |
| Source control | Git + GitHub                                                                                          |
| Secrets        | `python-dotenv` locally; Hugging Face Space "Secrets" in production                                   |

## 5. Out of Scope (For This Iteration)

- Multilingual support (English only for now)
- Streaming token-by-token responses
- Re-ranking with a cross-encoder
- Multi-tenant authentication
- Persistent conversation history across sessions
