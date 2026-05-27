# ============================================================
# main.py
# Servidor FastAPI - Agente de Suporte NOC - ISP
#
# Endpoints:
#   POST /chat          → Chat principal com o agente
#   GET  /health        → Health check do servidor
#   GET  /tools         → Lista as ferramentas disponíveis
#   DELETE /sessions/{session_id} → Limpa histórico de uma sessão
# ============================================================

import asyncio
import json
import logging
import secrets
import uuid as uuid_lib
import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent import run_agent, ALL_TOOLS
from llm_factory import LLM_ROUTER_PROVIDER, LLM_FORMATTER_PROVIDER

load_dotenv()

# ============================================================
# Configuração de Logging
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# Armazenamento de Sessões em Memória
# (Para produção, substitua por Redis ou banco de dados)
# ============================================================

# Formato: {session_id: {"history": [{"role": "user"|"assistant", "content": str}], "client_ip": str|None}}
session_store: dict[str, dict] = {}

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "50"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "1000"))

# Autenticação por API Key (opcional — se não configurada, aceita todas as origens internas)
_API_KEY = os.getenv("API_KEY", "")
_api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limiting simples por sessão (sem dependências extras)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))


# ============================================================
# Ciclo de Vida da Aplicação
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida do servidor FastAPI."""
    logger.info("=" * 60)
    logger.info("🚀 Agente de Suporte NOC iniciando...")
    _router_model = os.getenv("OLLAMA_MODEL") if LLM_ROUTER_PROVIDER == "ollama" else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    _fmt_model = os.getenv("OLLAMA_MODEL") if LLM_FORMATTER_PROVIDER == "ollama" else os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    logger.info("   Router LLM   : %s | %s", LLM_ROUTER_PROVIDER, _router_model)
    logger.info("   Formatter LLM: %s | %s", LLM_FORMATTER_PROVIDER, _fmt_model)
    logger.info("   Ferramentas registradas: %d", len(ALL_TOOLS))
    for tool in ALL_TOOLS:
        logger.info("   ✅ %s", tool.name)
    logger.info("=" * 60)
    yield
    logger.info("🛑 Agente de Suporte NOC encerrando...")


# ============================================================
# Inicialização do FastAPI
# ============================================================

app = FastAPI(
    title="Agente de Suporte NOC - ISP",
    description=(
        "API de atendimento ao cliente inteligente para Provedores de Internet. "
        "O agente de IA diagnostica problemas de conexão usando Zabbix e "
        "executa manutenções automáticas via Ansible."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: Requer configuração explícita de origens — nunca usa wildcard em produção
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
if not _cors_origins:
    logger.warning("CORS_ORIGINS não configurado — nenhuma origem cross-origin permitida.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Arquivos estáticos (CSS, JS, imagens futuros)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================
# Modelos Pydantic (Schemas da API)
# ============================================================

class ChatRequest(BaseModel):
    """Schema da requisição de chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Mensagem do cliente descrevendo seu problema.",
        examples=["Minha internet está muito lenta desde hoje de manhã."],
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "ID da sessão para manter o histórico da conversa. "
            "Se não fornecido, uma nova sessão é criada automaticamente."
        ),
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Olá, minha internet está caída desde as 10h.",
                "session_id": None,
            }
        }


class ChatResponse(BaseModel):
    """Schema da resposta do chat."""

    response: str = Field(description="Resposta do agente de suporte.")
    session_id: str = Field(description="ID da sessão (use nas próximas requisições).")
    message_count: int = Field(description="Total de mensagens na sessão atual.")
    timestamp: str = Field(description="Timestamp da resposta em ISO 8601.")


class HealthResponse(BaseModel):
    """Schema do health check."""

    status: str
    timestamp: str
    active_sessions: int
    tools_available: list[str]
    model: str


class ToolInfo(BaseModel):
    """Informações sobre uma ferramenta disponível."""

    name: str
    description: str
    category: str  # "zabbix" ou "ansible"


# ============================================================
# Segurança: Autenticação, Validação e Rate Limiting
# ============================================================

def _verify_api_key(key: str | None = Security(_api_key_scheme)) -> None:
    """Valida a API key se API_KEY estiver configurada no ambiente."""
    if not _API_KEY:
        return  # Sem API_KEY configurada: modo desenvolvimento (não expor em produção)
    if not key or not secrets.compare_digest(key, _API_KEY):
        raise HTTPException(status_code=403, detail="Acesso negado.")


def _is_valid_uuid(value: str) -> bool:
    """Verifica se a string é um UUID válido."""
    try:
        uuid_lib.UUID(str(value))
        return True
    except (ValueError, AttributeError):
        return False


def _check_rate_limit(session_id: str) -> bool:
    """Retorna True se dentro do limite; False se excedeu. Atualiza contadores."""
    session = session_store.get(session_id)
    if not session:
        return True

    now = datetime.now(timezone.utc)
    window_start = session.get("window_start", now)

    if (now - window_start).total_seconds() > RATE_LIMIT_WINDOW_SECONDS:
        session["req_count"] = 1
        session["window_start"] = now
        return True

    session["req_count"] = session.get("req_count", 0) + 1
    return session["req_count"] <= RATE_LIMIT_REQUESTS


# ============================================================
# Utilitário: Gestão de Sessão
# ============================================================

def get_or_create_session(session_id: str | None) -> tuple[str, list[dict], str | None, list[str]]:
    """
    Retorna uma sessão existente ou cria uma nova.

    Returns:
        Tupla (session_id, history, client_ip, actions_taken).
    """
    # Aceita apenas UUIDs válidos para session_id fornecidos pelo cliente
    if session_id and _is_valid_uuid(session_id) and session_id in session_store:
        session_data = session_store[session_id]
        if isinstance(session_data, dict):
            return (
                session_id,
                session_data.get("history", []),
                session_data.get("client_ip"),
                session_data.get("actions_taken", []),
            )

    # Cria nova sessão com UUID gerado pelo servidor
    new_id = str(uuid_lib.uuid4())

    if len(session_store) >= MAX_SESSIONS:
        oldest_key = next(iter(session_store))
        del session_store[oldest_key]
        logger.warning("Limite de sessões atingido. Sessão mais antiga removida.")

    session_store[new_id] = {
        "history": [],
        "client_ip": None,
        "actions_taken": [],
        "req_count": 0,
        "window_start": datetime.now(timezone.utc),
    }
    logger.info("Nova sessão criada: %s", new_id)
    return new_id, [], None, []


def trim_history(history: list[dict]) -> list[dict]:
    """
    Mantém o histórico dentro do limite configurado (MAX_HISTORY_MESSAGES).
    Remove as mensagens mais antigas preservando sempre o contexto mais recente.

    Args:
        history: Lista completa do histórico.

    Returns:
        Histórico aparado para o tamanho máximo.
    """
    if len(history) > MAX_HISTORY_MESSAGES:
        # Mantém as mensagens mais recentes (remove do início)
        trimmed = history[-MAX_HISTORY_MESSAGES:]
        logger.debug("Histórico aparado: %d → %d mensagens", len(history), len(trimmed))
        return trimmed
    return history


# ============================================================
# Middleware: Log de Requisições
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Loga todas as requisições com método, path e tempo de resposta."""
    start_time = datetime.now(timezone.utc)
    response = await call_next(request)
    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    logger.info(
        "%s %s → %d [%.0fms]",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ============================================================
# ENDPOINT RAIZ: GET / → Interface Web de Chat
# ============================================================

@app.get(
    "/",
    summary="Interface de Chat",
    description="Serve a interface web do Agente de Suporte.",
    tags=["Sistema"],
    include_in_schema=False,  # Não polui o Swagger
)
async def root() -> FileResponse:
    """Serve o HTML da interface de chat."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ============================================================
# ENDPOINT PRINCIPAL: POST /chat
# ============================================================

def _sse(event_type: str, **payload) -> str:
    """Formata um evento SSE."""
    return f"data: {json.dumps({'type': event_type, **payload})}\n\n"


@app.post(
    "/chat",
    summary="Enviar mensagem ao Agente de Suporte (SSE)",
    description="Retorna Server-Sent Events: 'status' imediato e 'done' com a resposta final.",
    tags=["Chat"],
)
async def chat(
    request: ChatRequest,
    _auth: None = Security(_verify_api_key),
) -> StreamingResponse:
    """Processa a mensagem e transmite o resultado via SSE para resposta imediata ao cliente."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="A mensagem não pode ser vazia.")

    session_id, history, client_ip, actions_taken = get_or_create_session(request.session_id)

    if not _check_rate_limit(session_id):
        logger.warning("Rate limit excedido para sessão: %s", session_id)
        raise HTTPException(status_code=429, detail="Muitas requisições. Aguarde um momento.")

    logger.info("Chat | session=%s | actions=%s | msg='%s...'", session_id, actions_taken, message[:60])

    async def generate():
        # Responde imediatamente — cliente vê feedback antes dos playbooks rodarem
        yield _sse("status", text="🔍 Verificando sua conexão, aguarde um momento...")

        agent_response = "Ocorreu um problema ao processar sua solicitação. Tente novamente."
        try:
            logger.info("Iniciando run_agent em thread para sessão '%s'", session_id)

            agent_response, updated_history, new_client_ip, updated_actions = await asyncio.to_thread(
                run_agent, message, history, client_ip, actions_taken
            )

            logger.info("run_agent concluído para sessão '%s'", session_id)

            existing = session_store.get(session_id, {})
            session_store[session_id] = {
                "history": trim_history(updated_history),
                "client_ip": new_client_ip or client_ip,
                "actions_taken": updated_actions,
                "req_count": existing.get("req_count", 1),
                "window_start": existing.get("window_start", datetime.now(timezone.utc)),
            }

        except Exception:
            logger.exception("Erro no run_agent para sessão '%s'", session_id)
            agent_response = "Erro interno. Por favor, tente novamente."

        finally:
            # Garante que o evento done SEMPRE é enviado, mesmo em caso de erro
            yield _sse(
                "done",
                response=agent_response,
                session_id=session_id,
                message_count=len((session_store.get(session_id) or {}).get("history", [])),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# ENDPOINT: GET /health
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check do Servidor",
    description="Verifica se o servidor está operacional e lista recursos disponíveis.",
    tags=["Sistema"],
)
async def health_check() -> HealthResponse:
    """Retorna o status de saúde do servidor."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        active_sessions=len(session_store),
        tools_available=[tool.name for tool in ALL_TOOLS],
        model=os.getenv("GROQ_MODEL", "llama3-70b-8192"),
    )


# ============================================================
# ENDPOINT: GET /tools
# ============================================================

@app.get(
    "/tools",
    response_model=list[ToolInfo],
    summary="Listar Ferramentas do Agente",
    description="Retorna todas as ferramentas disponíveis para o agente, com categoria e descrição.",
    tags=["Sistema"],
)
async def list_tools() -> list[ToolInfo]:
    """Lista todas as ferramentas registradas no agente."""
    from tools.zabbix import get_client_status, get_active_incidents, check_optical_signal

    zabbix_tool_names = {get_client_status.name, get_active_incidents.name, check_optical_signal.name}

    result = []
    for tool in ALL_TOOLS:
        category = "zabbix" if tool.name in zabbix_tool_names else "ansible"
        # Pega apenas a primeira linha da docstring como descrição curta
        short_desc = (tool.description or "").split("\n")[0].strip()
        result.append(ToolInfo(name=tool.name, description=short_desc, category=category))

    return result


# ============================================================
# ENDPOINT: DELETE /sessions/{session_id}
# ============================================================

@app.delete(
    "/sessions/{session_id}",
    summary="Limpar Histórico de Sessão",
    description="Remove o histórico de conversas de uma sessão específica.",
    tags=["Sessões"],
)
async def clear_session(
    session_id: str,
    _auth: None = Security(_verify_api_key),
) -> JSONResponse:
    """Remove o histórico de uma sessão."""
    if not _is_valid_uuid(session_id) or session_id not in session_store:
        raise HTTPException(status_code=404, detail="Sessão não encontrada.")

    del session_store[session_id]
    logger.info("Sessão removida: %s", session_id)

    return JSONResponse(
        content={
            "message": f"Sessão '{session_id}' removida com sucesso.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


# ============================================================
# Ponto de Entrada
# ============================================================

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("ENVIRONMENT", "production").lower() == "development"

    logger.info("Iniciando servidor em http://%s:%d", host, port)
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=LOG_LEVEL.lower(),
    )
