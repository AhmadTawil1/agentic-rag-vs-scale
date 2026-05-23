"""
System B (David): Agentic RAG with Self-Correction
Cyclic LangGraph pipeline with document grading, query rewriting, and
hallucination checking.  Uses Llama-3.1-8B via the Groq API.
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, List

sys.path.append(str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

from config import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    DAVID_TEMPERATURE,
    EMBEDDING_MODEL,
    MAX_GENERATION_RETRIES,
    MAX_RETRIEVAL_LOOPS,
    RETRIEVAL_TOP_K,
)

DAVID_MODEL = "llama-3.1-8b-instant"


class GraphState(TypedDict):
    question_id: str
    question: str
    query: str
    documents: List[Document]
    answer: str
    retrieval_attempts: int
    generation_attempts: int
    documents_relevant: bool
    answer_grounded: bool


def create_agentic_rag(
    vectorstore=None,
    collection_name: str | None = None,
    chroma_dir: str | None = None,
) -> object:
    """
    Build and compile the agentic RAG graph.

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

    llm = ChatGroq(
        model=DAVID_MODEL,
        temperature=DAVID_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    def retrieve(state: GraphState) -> dict:
        search_kwargs: dict = {"k": RETRIEVAL_TOP_K}
        if state.get("question_id"):
            search_kwargs["filter"] = {"question_id": state["question_id"]}
        documents = vectorstore.similarity_search(state["query"], **search_kwargs)
        return {
            "documents": documents,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }

    def grade_documents(state: GraphState) -> dict:
        doc_summaries = "\n".join(
            f"[{i+1}] {doc.page_content[:300]}" for i, doc in enumerate(state["documents"])
        )
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Documents:\n{doc_summaries}\n\n"
            "For each document, reply with its number followed by 'yes' or 'no' "
            "(e.g. '1: yes'). Is it relevant to the question?"
        )
        response = llm.invoke(prompt).content.lower()
        relevant_count = sum(
            1 for i in range(1, len(state["documents"]) + 1) if f"{i}: yes" in response
        )
        return {"documents_relevant": relevant_count >= len(state["documents"]) / 2}

    def transform_query(state: GraphState) -> dict:
        prompt = (
            f"Rewrite this search query to retrieve better results.\n\n"
            f"Original question: {state['question']}\n"
            f"Current query: {state['query']}\n\n"
            "Provide ONLY the rewritten query, nothing else."
        )
        return {"query": llm.invoke(prompt).content.strip()}

    def generate(state: GraphState) -> dict:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = (
            "Use the following context to answer the question. "
            "If the answer is not present, say "
            "'I don't have enough information to answer this question.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {state['question']}\n\nAnswer:"
        )
        return {
            "answer": llm.invoke(prompt).content.strip(),
            "generation_attempts": state.get("generation_attempts", 0) + 1,
        }

    def check_hallucination(state: GraphState) -> dict:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = (
            f"Context:\n{context}\n\n"
            f"Answer: {state['answer']}\n\n"
            "Is every factual claim in the answer supported by the context above? "
            "Answer ONLY 'yes' or 'no'."
        )
        return {"answer_grounded": "yes" in llm.invoke(prompt).content.strip().lower()}

    def route_after_grading(state: GraphState) -> str:
        if state["documents_relevant"]:
            return "generate"
        if state["retrieval_attempts"] < MAX_RETRIEVAL_LOOPS:
            return "transform_query"
        return "generate"

    def route_after_check(state: GraphState) -> str:
        if state["answer_grounded"]:
            return END
        if state["generation_attempts"] < MAX_GENERATION_RETRIES:
            return "generate"
        return END

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("generate", generate)
    workflow.add_node("check_hallucination", check_hallucination)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"transform_query": "transform_query", "generate": "generate"},
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("generate", "check_hallucination")
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {"generate": "generate", END: END},
    )

    return workflow.compile()


def run_agentic_rag(
    question: str,
    question_id: str | None = None,
    app=None,
    vectorstore=None,
    collection_name: str | None = None,
    chroma_dir: str | None = None,
) -> dict:
    """
    Run the agentic RAG pipeline on a single question.

    Pass a pre-compiled `app` (from create_agentic_rag) and a pre-initialised
    `vectorstore` to avoid rebuilding the graph on every call.
    """
    if app is None:
        app = create_agentic_rag(
            vectorstore=vectorstore,
            collection_name=collection_name,
            chroma_dir=chroma_dir,
        )

    final_state = app.invoke({
        "question_id": question_id or "",
        "question": question,
        "query": question,
        "documents": [],
        "answer": "",
        "retrieval_attempts": 0,
        "generation_attempts": 0,
        "documents_relevant": False,
        "answer_grounded": False,
    })

    return {
        "question": question,
        "answer": final_state["answer"],
        "retrieval_attempts": final_state["retrieval_attempts"],
        "generation_attempts": final_state["generation_attempts"],
        "documents": final_state["documents"],
    }


if __name__ == "__main__":
    result = run_agentic_rag("What is retrieval augmented generation?")
    print(f"Answer: {result['answer']}")
    print(f"Retrieval attempts: {result['retrieval_attempts']}")
    print(f"Generation attempts: {result['generation_attempts']}")
