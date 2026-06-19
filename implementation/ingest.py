import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """Load markdown files from each top-level folder under knowledge-base/."""
    documents = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        if not os.path.isdir(folder):
            continue
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    """Split documents into overlapping chunks suitable for retrieval.

    Each chunk is prefixed with its source document title (derived from the
    filename) so that retrieval retains document-level context even when a
    section heading like "## Key Features" does not repeat the product name.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        title = Path(chunk.metadata.get("source", "")).stem
        if title:
            chunk.page_content = f"[{title}]\n\n{chunk.page_content}"
    return chunks


def build_vectorstore(chunks):
    """Create (or overwrite) the Chroma vector store."""
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    count = vectorstore._collection.count()
    sample = vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"]
    dims = len(sample[0]) if len(sample) else 0
    print(f"Vectorstore ready: {count:,} vectors, {dims:,} dimensions")
    return vectorstore


if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    build_vectorstore(chunks)
    print("Ingestion complete")
