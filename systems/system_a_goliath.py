"""
System A (Goliath): Baseline Naive RAG
Linear pipeline: Query → Retrieve → Generate.  Uses GPT-5.4-mini with no self-correction.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GOLIATH_MODEL,
    GOLIATH_TEMPERATURE,
    RETRIEVAL_TOP_K,
)


def run_baseline_rag(
    question: str,
    question_id: str | None = None,
    vectorstore=None,
    collection_name: str | None = None,
    chroma_dir: str | None = None,
) -> dict:
    """
    Run the baseline naive RAG pipeline on a single question.

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

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    llm = ChatOpenAI(model=GOLIATH_MODEL, temperature=GOLIATH_TEMPERATURE)

    documents = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = (
        "Use the following context to answer the question. "
        "If the answer is not present in the context, say "
        "'I don't have enough information to answer this question.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    answer = llm.invoke(prompt).content

    return {
        "question": question,
        "answer": answer,
        "source_documents": documents,
    }


if __name__ == "__main__":
    result = run_baseline_rag("What is retrieval augmented generation?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['source_documents'])} documents retrieved")
