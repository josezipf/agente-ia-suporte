# ============================================================
# llm_factory.py
# Fábrica de modelos LLM — providers independentes por função
#
# Router  (decisão JSON):   LLM_ROUTER_PROVIDER    (padrão: ollama)
# Formatter (texto amigável): LLM_FORMATTER_PROVIDER (padrão: groq)
#
# Separar os dois evita alucinação do formatter sem sobrecarregar
# a GPU local com tarefas de geração de texto mais complexas.
# ============================================================

import os
import logging
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

logger = logging.getLogger(__name__)

# Provider do decisor JSON — modelo local (rápido, tarefa simples)
LLM_ROUTER_PROVIDER = os.getenv("LLM_ROUTER_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")).lower()

# Provider do formatter de texto — modelo confiável (sem alucinação)
LLM_FORMATTER_PROVIDER = os.getenv("LLM_FORMATTER_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")).lower()

# Compatibilidade retroativa: LLM_PROVIDER define ambos se os específicos não forem setados
LLM_PROVIDER = LLM_ROUTER_PROVIDER  # usado só para log de startup

_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", os.getenv("GROQ_TEMPERATURE", "0.0")))


def _create_ollama(max_tokens: int) -> BaseChatModel:
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError("langchain-ollama não instalado. Execute: pip install langchain-ollama")

    model = os.getenv("OLLAMA_MODEL", "gemma3:latest")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    logger.debug("LLM Ollama | model=%s | base_url=%s", model, base_url)
    return ChatOllama(model=model, base_url=base_url, temperature=_TEMPERATURE, num_predict=max_tokens)


def _create_groq(max_tokens: int) -> BaseChatModel:
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("langchain-groq não instalado. Execute: pip install langchain-groq")

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    api_key = os.getenv("GROQ_API_KEY", "")
    logger.debug("LLM Groq | model=%s", model)
    return ChatGroq(api_key=api_key, model=model, temperature=_TEMPERATURE, max_tokens=max_tokens)


def _build(provider: str, max_tokens: int) -> BaseChatModel:
    if provider == "ollama":
        return _create_ollama(max_tokens)
    return _create_groq(max_tokens)


def create_router_llm(max_tokens: int) -> BaseChatModel:
    """Modelo para decisão JSON — usa LLM_ROUTER_PROVIDER (padrão: ollama/gemma3)."""
    llm = _build(LLM_ROUTER_PROVIDER, max_tokens)
    logger.debug("Router LLM: %s | %s", LLM_ROUTER_PROVIDER, type(llm).__name__)
    return llm


def create_formatter_llm(max_tokens: int) -> BaseChatModel:
    """Modelo para formatação de texto — usa LLM_FORMATTER_PROVIDER (padrão: groq)."""
    llm = _build(LLM_FORMATTER_PROVIDER, max_tokens)
    logger.debug("Formatter LLM: %s | %s", LLM_FORMATTER_PROVIDER, type(llm).__name__)
    return llm


# Mantém compatibilidade com código que chame create_llm diretamente
def create_llm(max_tokens: int) -> BaseChatModel:
    return _build(LLM_ROUTER_PROVIDER, max_tokens)
