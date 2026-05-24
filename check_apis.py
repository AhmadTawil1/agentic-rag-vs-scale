"""
Quick connectivity check for all LLM backends used in the project.
Tests: GROQ_API_KEY, GROQ_API_KEY_2, and local Ollama.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_BASE = "http://localhost:11434"
PROBE = "Reply with the single word: OK"


def check_groq(key_name: str) -> None:
    key = os.getenv(key_name)
    if not key:
        print(f"  [{key_name}]  SKIP — not set in .env")
        return
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=key)
        reply = llm.invoke(PROBE).content.strip()
        print(f"  [{key_name}]  OK  ->  '{reply}'")
    except Exception as e:
        print(f"  [{key_name}]  FAIL — {e}")


def check_ollama() -> None:
    try:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0, base_url=OLLAMA_BASE)
        reply = llm.invoke(PROBE).content.strip()
        print(f"  [Ollama local ({OLLAMA_MODEL})]  OK  ->  '{reply}'")
    except ImportError:
        print("  [Ollama local]  SKIP — langchain-ollama not installed  (pip install langchain-ollama)")
    except Exception as e:
        print(f"  [Ollama local ({OLLAMA_MODEL})]  FAIL — {e}")


if __name__ == "__main__":
    print("\n=== LLM backend connectivity check ===\n")
    check_groq("GROQ_API_KEY")
    check_groq("GROQ_API_KEY_2")
    check_ollama()
    print()
