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

import httpx
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
    # Configura webhook do Telegram se as variáveis existirem
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_webhook = os.getenv("TELEGRAM_WEBHOOK_URL")
    if tg_token and tg_webhook:
        webhook_target = f"{tg_webhook.rstrip('/')}/webhook/telegram/{tg_token}"
        logger.info("🤖 Configurando Webhook do Telegram para: %s", webhook_target)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"https://api.telegram.org/bot{tg_token}/setWebhook",
                    json={"url": webhook_target},
                    timeout=10.0
                )
                resp_json = resp.json()
                if resp_json.get("ok"):
                    logger.info("✅ Webhook do Telegram registrado com sucesso!")
                else:
                    logger.error("❌ Falha ao registrar Webhook: %s", resp_json.get("description"))
        except Exception as e:
            logger.error("❌ Erro ao conectar na API do Telegram: %s", e)
    else:
        logger.info("ℹ️ Telegram não configurado (TELEGRAM_BOT_TOKEN / TELEGRAM_WEBHOOK_URL ausentes)")
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


def get_or_create_external_session(session_id: str) -> tuple[str, list[dict], str | None, list[str]]:
    """
    Retorna uma sessão externa existente (ex: Telegram/WhatsApp) ou cria uma nova.
    Evita a necessidade de validação por UUID para IDs vindos de APIs externas.
    """
    if session_id in session_store:
        session_data = session_store[session_id]
        if isinstance(session_data, dict):
            return (
                session_id,
                session_data.get("history", []),
                session_data.get("client_ip"),
                session_data.get("actions_taken", []),
            )

    if len(session_store) >= MAX_SESSIONS:
        oldest_key = next(iter(session_store))
        del session_store[oldest_key]
        logger.warning("Limite de sessões atingido. Sessão mais antiga removida.")

    session_store[session_id] = {
        "history": [],
        "client_ip": None,
        "actions_taken": [],
        "req_count": 0,
        "window_start": datetime.now(timezone.utc),
    }
    logger.info("Nova sessão externa criada: %s", session_id)
    return session_id, [], None, []


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




# ============================================================
# ENDPOINT: POST /webhook/telegram/{token}
# ============================================================

async def _send_telegram_message(token: str, chat_id: int, text: str):
    """Envia uma mensagem de texto para o chat do Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json={"chat_id": chat_id, "text": text}, timeout=10.0)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Erro ao enviar mensagem ao Telegram: %s", e)


async def _send_telegram_typing(token: str, chat_id: int):
    """Envia a ação de 'digitando...' ao chat do Telegram."""
    url = f"https://api.telegram.org/bot{token}/sendChatAction"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"chat_id": chat_id, "action": "typing"}, timeout=5.0)
        except Exception:
            pass  # Se falhar, não impede o bot de funcionar


@app.post(
    "/webhook/telegram/{token}",
    summary="Receber mensagens do Telegram",
    description="Endpoint de webhook para processar mensagens enviadas pelo Telegram.",
    tags=["Webhooks"],
)
async def telegram_webhook(token: str, request: Request):
    """Recebe mensagens do Telegram, processa com o Agente e responde."""
    env_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not env_token or token != env_token:
        raise HTTPException(status_code=403, detail="Acesso negado.")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Payload inválido.")

    # Filtra apenas mensagens com texto
    if "message" not in payload or "text" not in payload["message"]:
        return {"status": "ignored"}

    chat_id = payload["message"]["chat"]["id"]
    user_message = payload["message"]["text"].strip()
    session_id = f"telegram_{chat_id}"

    if not user_message:
        return {"status": "ignored"}

    # Resgata ou inicia a sessão externa do Telegram
    _, history, client_ip, actions_taken = get_or_create_external_session(session_id)

    # Avisa ao Telegram que o bot está "digitando..."
    await _send_telegram_typing(token, chat_id)

    try:
        # Roda o agente NOC em thread para não travar o loop assíncrono
        agent_response, updated_history, new_client_ip, updated_actions = await asyncio.to_thread(
            run_agent, user_message, history, client_ip, actions_taken
        )

        # Atualiza a sessão
        existing = session_store.get(session_id, {})
        session_store[session_id] = {
            "history": trim_history(updated_history),
            "client_ip": new_client_ip or client_ip,
            "actions_taken": updated_actions,
            "req_count": existing.get("req_count", 1),
            "window_start": existing.get("window_start", datetime.now(timezone.utc)),
        }

        # Envia a resposta final gerada pelo agente
        await _send_telegram_message(token, chat_id, agent_response)

    except Exception:
        logger.exception("Erro ao processar mensagem do Telegram para chat_id %s", chat_id)
        await _send_telegram_message(token, chat_id, "Desculpe, ocorreu um erro interno ao processar sua solicitação.")

    return {"status": "ok"}





# ============================================================
# ENDPOINT: GET & POST /webhook/whatsapp
# ============================================================

async def _send_whatsapp_message(token: str, phone_number_id: str, recipient: str, text: str):
    """Envia uma mensagem de texto usando a API de Nuvem do WhatsApp (Meta)."""
    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {"body": text},
    }
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=10.0)
            resp.raise_for_status()
            logger.info("Mensagem enviada com sucesso para o WhatsApp: %s", recipient)
        except Exception as e:
            logger.error("Erro ao enviar mensagem ao WhatsApp: %s", e)


async def _mark_whatsapp_read(token: str, phone_number_id: str, message_id: str):
    """Marca uma mensagem recebida no WhatsApp como lida (check azul)."""
    url = f"https://graph.facebook.com/v21.0/{phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json=payload, headers=headers, timeout=5.0)
        except Exception:
            pass  # Falhar no status de leitura não interrompe o bot


async def _process_and_reply_whatsapp(
    token: str,
    phone_number_id: str,
    session_id: str,
    recipient: str,
    message: str,
    history: list[dict],
    client_ip: str | None,
    actions_taken: list[str]
):
    """Processa a mensagem e envia a resposta de volta ao WhatsApp (background task)."""
    try:
        agent_response, updated_history, new_client_ip, updated_actions = await asyncio.to_thread(
            run_agent, message, history, client_ip, actions_taken
        )

        # Atualiza a sessão
        existing = session_store.get(session_id, {})
        session_store[session_id] = {
            "history": trim_history(updated_history),
            "client_ip": new_client_ip or client_ip,
            "actions_taken": updated_actions,
            "req_count": existing.get("req_count", 1),
            "window_start": existing.get("window_start", datetime.now(timezone.utc)),
        }

        # Envia a resposta final gerada pelo agente
        await _send_whatsapp_message(token, phone_number_id, recipient, agent_response)

    except Exception:
        logger.exception("Erro no processamento em background do WhatsApp para %s", recipient)
        await _send_whatsapp_message(token, phone_number_id, recipient, "Desculpe, ocorreu um erro interno ao processar sua solicitação.")


@app.get(
    "/webhook/whatsapp",
    summary="Validar Webhook do WhatsApp",
    description="Endpoint GET exigido pela Meta para verificar o webhook do WhatsApp.",
    tags=["Webhooks"],
)
async def verify_whatsapp_webhook(request: Request):
    """Valida o webhook do WhatsApp respondendo ao desafio (challenge) da Meta."""
    params = request.query_params
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN")

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == verify_token:
        logger.info("✅ Webhook do WhatsApp validado com sucesso pela Meta!")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(challenge)

    logger.warning("❌ Falha na validação do webhook do WhatsApp. Token incorreto.")
    raise HTTPException(status_code=403, detail="Token de verificação inválido.")


@app.post(
    "/webhook/whatsapp",
    summary="Receber mensagens do WhatsApp",
    description="Endpoint POST para processar as mensagens enviadas pelos clientes via WhatsApp.",
    tags=["Webhooks"],
)
async def whatsapp_webhook(request: Request):
    """Recebe mensagens do WhatsApp, dispara a resposta em background e responde 200 OK imediatamente."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Payload inválido.")

    # Filtra as mensagens recebidas
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})

        if "messages" not in value:
            return {"status": "ignored"}

        message_data = value["messages"][0]
        msg_type = message_data.get("type")

        # Apenas processa mensagens do tipo texto
        if msg_type != "text":
            return {"status": "ignored"}

        from_number = message_data.get("from")
        message_id = message_data.get("id")
        user_message = message_data.get("text", {}).get("body", "").strip()

        if not from_number or not user_message:
            return {"status": "ignored"}

        session_id = f"whatsapp_{from_number}"
        
        token = os.getenv("WHATSAPP_ACCESS_TOKEN")
        phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

        if not token or not phone_number_id:
            logger.error("WhatsApp não configurado. Defina WHATSAPP_ACCESS_TOKEN e WHATSAPP_PHONE_NUMBER_ID no .env")
            return {"status": "error", "message": "WhatsApp credentials not configured"}

        # Resgata ou cria sessão externa
        _, history, client_ip, actions_taken = get_or_create_external_session(session_id)

        # Marca a mensagem como lida (check azul)
        await _mark_whatsapp_read(token, phone_number_id, message_id)

        # Dispara o processamento em background (evita estourar o timeout de 3s da Meta)
        asyncio.create_task(
            _process_and_reply_whatsapp(
                token=token,
                phone_number_id=phone_number_id,
                session_id=session_id,
                recipient=from_number,
                message=user_message,
                history=history,
                client_ip=client_ip,
                actions_taken=actions_taken
            )
        )

    except Exception:
        logger.exception("Erro ao processar webhook do WhatsApp")

    return {"status": "ok"}


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
