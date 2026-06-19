---
title: Iziko RAG Chatbot
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
license: mit
---

# Iziko Cloud Solutions - RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
**Iziko Cloud Solutions**, a fictional South African cloud and IT services
company headquartered in Cape Town.

The corpus is a small, hand-authored knowledge base of company information,
product specifications, employee bios, and enterprise customer contracts. The
chatbot uses LangChain + Chroma to retrieve relevant context and an OpenAI
chat model to ground its responses in that retrieved content.

> **Note:** Iziko Cloud Solutions, its employees, contracts, and customers
> are entirely fictional. This project was built as a coursework assignment
> based on the structure of a training-session reference project.

## Live Demo

- Hugging Face Space: <https://huggingface.co/spaces/Olefile/iziko-rag-chatbot>
- GitHub repository: <https://github.com/OlefileMamorare/iziko-rag-chatbot>

## Repository Structure

```
iziko-rag-chatbot/
├── app.py                       # Gradio chat UI (entry point on HF Spaces)
├── main.py                      # Trivial CLI entry point
├── evaluator.py                 # Gradio evaluation dashboard
├── pyproject.toml               # uv / pip project metadata
├── requirements.txt             # Pinned-ish runtime dependencies
├── .env.example                 # Template for required env vars
├── PLANNING.md                  # Project planning document (deliverable #1)
├── README.md                    # You are here
├── implementation/
│   ├── ingest.py                # Load docs, chunk, embed, persist to Chroma
│   └── answer.py                # Retrieval + LLM answer with history
├── evaluation/
│   ├── eval.py                  # Retrieval + LLM-as-a-judge evaluation
│   ├── test.py                  # TestQuestion model + loader
│   └── tests.jsonl              # 28 evaluation questions
└── knowledge-base/
    ├── company/                 # About, overview, culture, careers
    ├── products/                # IzikoConnect, IzikoGuard, IzikoData, IzikoFlow
    ├── employees/               # Bios for key employees
    └── contracts/               # Enterprise customer contracts
```

## Prerequisites

- Python 3.11 or newer
- An OpenAI API key (used for both embeddings and the chat model)

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/iziko-rag-chatbot.git
cd iziko-rag-chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# then edit .env and paste your OPENAI_API_KEY

# 5. Build the vector store (one-off; rerun whenever knowledge-base changes)
python -m implementation.ingest

# 6. Launch the chatbot
python app.py
```

Gradio will print a local URL (usually <http://127.0.0.1:7860>).

## Running the Evaluation Dashboard

```bash
python evaluator.py
```

This launches a separate Gradio app where you can run:

- **Retrieval evaluation:** MRR, nDCG@10, keyword coverage
- **Answer evaluation:** LLM-as-a-judge scoring on accuracy, completeness, relevance

You can also evaluate a single test from the CLI:

```bash
python -m evaluation.eval 0    # evaluate test row 0
```

## How It Works

1. **Ingestion.** All `.md` files under [knowledge-base/](knowledge-base) are
   loaded, split into 500-character chunks with 200-character overlap, embedded
   with `text-embedding-3-large`, and persisted to a local Chroma store under
   `vector_db/` (gitignored).
2. **Retrieval.** Each user question (optionally combined with the previous
   user turn) is used to retrieve the top **5** chunks from Chroma.
3. **Generation.** A system prompt instructing the model to act as the Iziko
   assistant is combined with the retrieved context, the chat history, and
   the new user message, then sent to `gpt-4.1-nano` with `temperature=0`.
4. **Display.** Both the answer and the underlying retrieved chunks are shown
   in the Gradio UI so users can verify the source.

See [PLANNING.md](PLANNING.md) for the full architecture diagram and design rationale.

## Deploying to Hugging Face Spaces

The chatbot is designed to deploy as-is to a free Hugging Face Space using the
Gradio SDK.

### Step 1 - Create the Space

1. Sign in at <https://huggingface.co/>.
2. Click **New** -> **Space**.
3. Configure:
   - **Space name:** `iziko-rag-chatbot`
   - **License:** MIT
   - **SDK:** Gradio
   - **Hardware:** CPU basic (free) is sufficient
4. Click **Create Space**.

### Step 2 - Add your OpenAI key as a secret

In the Space's **Settings -> Variables and secrets** panel, add:

- Name: `OPENAI_API_KEY`
- Value: your OpenAI API key
- Type: **Secret**

The app reads this via `os.environ` (loaded through `python-dotenv`), so no
code changes are needed.

### Step 3 - Push the code

Hugging Face Spaces are backed by a Git repository. From the project root:

```bash
# Add the Space as a second remote (one-off setup)
git remote add space https://huggingface.co/spaces/<your-hf-username>/iziko-rag-chatbot

# Make sure the YAML header at the top of README.md is committed
git add .
git commit -m "Initial commit"

# Push to Hugging Face
git push space main
```

If your Space defaults to a different branch name, use `git push space main:main`
or rename your local branch accordingly.

### Step 4 - First-build notes

- The first build will install everything in [requirements.txt](requirements.txt)
  and start `app.py`. The Space will be available at
  `https://huggingface.co/spaces/<your-hf-username>/iziko-rag-chatbot`.
- The vector store is **not** committed to the repo (`vector_db/` is in
  [.gitignore](.gitignore)). On a fresh Space, the app will need to build it
  on first run. You can either:
  - (a) Run `python -m implementation.ingest` once via the Space's "Logs"
    terminal / a small startup hook, **or**
  - (b) Remove `vector_db/` from `.gitignore`, run ingestion locally, and
    commit the resulting Chroma store so the Space starts with it preloaded
    (recommended for free hardware).

## Continuous Development

This project is committed to GitHub regularly. Commits should be small and
descriptive, e.g.:

- `feat: add IzikoGuard product page`
- `fix: handle empty history in resolve_query`
- `chore: bump gradio to 4.44`
- `eval: add spanning question about Pioneer Retail`

## License

MIT. See [LICENSE](LICENSE) if present, otherwise the MIT terms in the YAML
header above.

## Credits

This project was built as a personal RAG coursework assignment, using the
training-session reference project (an insurance-domain RAG) as a structural
guide. All company, product, employee, and contract content for Iziko Cloud
Solutions is fictional and was authored for this exercise.
