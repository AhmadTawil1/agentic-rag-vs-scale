"""
LLM factory for Llama systems.

Backend selection (checked in order):
  1. USE_OLLAMA=true      -> Ollama running locally or on Colab (no HF token needed)
  2. USE_LOCAL_LLM=true   -> HuggingFace 4-bit on GPU (needs HF_TOKEN + Meta license)
  3. GROQ_API_KEY set     -> Groq API with 3-tier fallback (key1 -> key2 -> Ollama)

Local models are cached as singletons — loaded only once per process.
"""

import os
from functools import lru_cache

from config import (
    DAVID_MODEL,
    DAVID_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_LLAMA_MODEL,
)


# ---------------------------------------------------------------------------
# Groq invoke with 3-tier fallback
# ---------------------------------------------------------------------------

def groq_invoke(llm, prompt: str, model: str, temperature: float):
    """Invoke a ChatGroq LLM: key-1 -> key-2 -> local Ollama on rate limit."""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        if "rate" not in str(e).lower():
            raise
        secondary = os.getenv("GROQ_API_KEY_2")
        if secondary:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(model=model, temperature=temperature, api_key=secondary).invoke(prompt)
            except Exception:
                pass
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=OLLAMA_LLAMA_MODEL,
                temperature=temperature,
                base_url=OLLAMA_BASE_URL,
            ).invoke(prompt)
        except Exception as ollama_err:
            raise RuntimeError(
                f"All Groq keys rate-limited and local Ollama unavailable: {ollama_err}"
            ) from e


# ---------------------------------------------------------------------------
# Backend builders
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_ollama_llm():
    """Connect to Ollama (local or Colab). Cached — one connection per process."""
    from langchain_ollama import ChatOllama
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    model = os.getenv("OLLAMA_MODEL", OLLAMA_LLAMA_MODEL)
    print(f"[llm_factory] Using Ollama  model={model}  base_url={base_url}")
    return ChatOllama(model=model, temperature=0, base_url=base_url)


@lru_cache(maxsize=1)
def _build_local_llm():
    """Load model in 4-bit on GPU via HuggingFace. Cached — loads only once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    hf_token = os.getenv("HF_TOKEN")
    model_id = os.getenv("LOCAL_LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    print(f"[llm_factory] Loading {model_id} in 4-bit on GPU (runs once)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto", token=hf_token
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
    print("[llm_factory] Model loaded and ready.")
    return llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_llama_llm(model: str = DAVID_MODEL, temperature: float = DAVID_TEMPERATURE):
    """
    Return the appropriate Llama LLM based on environment variables:
      USE_OLLAMA=true    -> Ollama (Colab or local, no HF token needed)
      USE_LOCAL_LLM=true -> HuggingFace 4-bit GPU (needs HF_TOKEN)
      default            -> Groq API
    """
    if os.getenv("USE_OLLAMA", "").lower() == "true":
        return _build_ollama_llm()

    if os.getenv("USE_LOCAL_LLM", "").lower() == "true":
        return _build_local_llm()

    from langchain_groq import ChatGroq
    key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY_2")
    if not key:
        raise ValueError("No LLM backend configured. Set USE_OLLAMA=true, USE_LOCAL_LLM=true, or GROQ_API_KEY.")
    return ChatGroq(model=model, temperature=temperature, api_key=key)


def invoke_llama(llm, prompt: str, model: str = DAVID_MODEL, temperature: float = DAVID_TEMPERATURE):
    """
    Invoke the LLM. ChatGroq gets the 3-tier rate-limit fallback;
    Ollama and local models are called directly.
    """
    from langchain_groq import ChatGroq
    if isinstance(llm, ChatGroq):
        return groq_invoke(llm, prompt, model, temperature)
    return llm.invoke(prompt)
