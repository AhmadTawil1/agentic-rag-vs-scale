"""
System C (Hermes): Naive RAG + Llama-3.1-8B
Linear pipeline: Query -> Retrieve -> Generate.  No self-correction.

LLM backend (auto-selected by llm_factory):
  - Local GPU (Colab T4): when USE_LOCAL_LLM=true  -> no rate limits
  - Groq API:             otherwise                 -> GROQ_API_KEY with 3-tier fallback
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    HERMES_MODEL,
    HERMES_TEMPERATURE,
    RETRIEVAL_TOP_K,
)
from systems.llm_factory import get_llama_llm, invoke_llama


def run_naive_llama_rag(
    question: str,
    question_id: str | None = None,
    vectorstore=None,
    collection_name: str | None = None,
    chroma_dir: str | None = None,
) -> dict:
    """
    Run the naive Llama RAG pipeline on a single question.

    Pass a pre-initialised `vectorstore` to avoid reloading the embedding model
    on every call (critical when running 200+ questions in a tournament).
    """
    if vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vectorstore = Chroma(
            persist_directory=chroma_dir or CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=collection_name or COLLECTION_NAME,
        )

    search_kwargs: dict = {"k": RETRIEVAL_TOP_K}
    if question_id:
        search_kwargs["filter"] = {"question_id": question_id}

    documents = vectorstore.as_retriever(search_kwargs=search_kwargs).invoke(question)

    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not present in the context, say "
        "'I don't have enough information to answer this question.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    llm = get_llama_llm(model=HERMES_MODEL, temperature=HERMES_TEMPERATURE)
    answer = invoke_llama(llm, prompt, HERMES_MODEL, HERMES_TEMPERATURE).content

    return {
        "question": question,
        "answer": answer,
        "source_documents": documents,
    }


if __name__ == "__main__":
    result = run_naive_llama_rag("What is retrieval augmented generation?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['source_documents'])} documents retrieved")
