# ============================================================
# agent.py
# Agente de Suporte NOC - ISP
#
# Arquitetura: JSON Router
#   - LLM decide APENAS qual ação executar (JSON)
#   - Execução é 100% determinística via ferramentas Python / Ansible
#   - Sem encadeamento autônomo de ações
# ============================================================

import os
import re
import json
import logging

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool
from dotenv import load_dotenv

from llm_factory import create_router_llm, create_formatter_llm, LLM_ROUTER_PROVIDER, LLM_FORMATTER_PROVIDER
from tools.zabbix import get_client_status, get_active_incidents, check_optical_signal
from tools.ansible import run_router_diagnostic, reset_pppoe_connection, reboot_client_cpe

load_dotenv()

# RateLimitError é específica do Groq — importa condicionalmente para não quebrar no Ollama
try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    GroqRateLimitError = None

logger = logging.getLogger(__name__)

# ============================================================
# Configuração
# ============================================================

MAX_TOKENS_ROUTER = int(os.getenv("LLM_MAX_TOKENS_ROUTER", os.getenv("GROQ_MAX_TOKENS_ROUTER", "256")))
MAX_TOKENS_FORMATTER = int(os.getenv("LLM_MAX_TOKENS_FORMATTER", os.getenv("GROQ_MAX_TOKENS_FORMATTER", "600")))
# Limite de mensagens do histórico enviadas ao LLM (economia de tokens)
MAX_HISTORY_TO_LLM = int(os.getenv("MAX_HISTORY_TO_LLM", "6"))

ZABBIX_TOOLS: list[BaseTool] = [get_client_status, get_active_incidents, check_optical_signal]
ANSIBLE_TOOLS: list[BaseTool] = [run_router_diagnostic, reset_pppoe_connection, reboot_client_cpe]
ALL_TOOLS: list[BaseTool] = ZABBIX_TOOLS + ANSIBLE_TOOLS

# ============================================================
# Validação de Entrada
# ============================================================

# Permite letras, números, espaços e separadores comuns em nomes/contratos/CPF/tel
_SAFE_ID_RE = re.compile(r'^[\w\s\-\.\@\/\(\)]+$')
# IP v4 válido
_VALID_IP_RE = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')
# Prefixos que o cliente usa ao se identificar: "meu nome é", "me chamo", etc.
_ID_PREFIX_RE = re.compile(
    r'^(?:meu\s+nome\s+(?:completo\s+)?[eé]\s*:?\s*'
    r'|me\s+chamo\s+'
    r'|sou\s+(?:o\s+|a\s+)?'
    r'|meu\s+(?:n[uú]mero\s+de\s+)?contrato\s+[eé]\s*:?\s*'
    r'|(?:n[uú]mero\s+de\s+)?contrato\s*:?\s*'
    r'|cpf\s*:?\s*)',
    re.IGNORECASE,
)


def _clean_identifier(value: str) -> str:
    """Remove prefixos conversacionais, deixando só o identificador."""
    return _ID_PREFIX_RE.sub("", value).strip()


def _validate_identifier(value: str) -> bool:
    """Verifica se o identificador do cliente é seguro (sem injeção).

    Rejeita frases longas para evitar que o LLM passe a mensagem completa
    do cliente como identificador (ex: 'Estou sem internet não consigo acessar').
    Nomes brasileiros raramente ultrapassam 5 palavras.
    """
    if not value or not (2 <= len(value) <= 100) or not _SAFE_ID_RE.match(value):
        return False
    if len(value.split()) > 5:
        return False
    return True


def _validate_ip(value: str) -> bool:
    """Verifica se o IP é um IPv4 válido."""
    if not value or not _VALID_IP_RE.match(value):
        return False
    return all(0 <= int(p) <= 255 for p in value.split('.'))


def _has_active_incidents(incidents_text: str) -> bool:
    """Retorna True se o Zabbix reportou pelo menos um incidente ativo."""
    t = incidents_text.lower()
    return "incidente" in t and ("ativo" in t or "🚨" in incidents_text)


# ============================================================
# Prompts — compactos para economizar tokens
# ============================================================

ROUTER_PROMPT = """Você é ALEX, decisor de suporte NOC de um ISP.
Sua única função é analisar a conversa e retornar um JSON. Nada mais.

FORMATO DE SAÍDA — retorne EXATAMENTE este JSON, sem texto antes ou depois:
{"action":"string","params":{"identifier":"string","client_ip":"string"},"message":"string"}

PROIBIÇÕES ABSOLUTAS:
- JAMAIS peça endereço IP ao cliente. O sistema busca o IP automaticamente pelo nome.
- JAMAIS peça dados técnicos ao cliente (IP, MAC, roteador, modelo, etc.).
- JAMAIS invente identificador. Use somente o que o cliente digitou.
- Campos não usados = string vazia "", NUNCA null.
- Nunca cite: Zabbix, Ansible, PPPoE, SSH, trigger, playbook.

ÁRVORE DE DECISÃO — siga os passos em ordem:

PASSO 1: O cliente apenas cumprimentou (ex: "Oi", "Bom dia", "Tudo bem?") e AINDA NÃO explicou o problema?
  → SIM → action = "message"
          message = "Olá! Sou o Alex, assistente virtual do suporte técnico. Como posso te ajudar hoje?"
          PARE AQUI.

PASSO 2: O cliente relatou o problema, mas AINDA NÃO informou o nome completo?
  → SIM → action = "ask_identification"
          message = "Compreendo a situação. Para que eu possa verificar o seu sinal, por favor, me informe o seu nome completo ou número do contrato."
          PARE AQUI.

PASSO 3: O cliente informou o nome, mas o contexto NÃO mostra "[Sessão: IP verificado = X.X.X.X]"?
  → SIM → action = "diagnose"
          params.identifier = somente o nome que o cliente informou (ex: "João da Silva")
          PARE AQUI. Não peça mais nada ao cliente.

PASSO 4: O cliente já passou pelo diagnóstico e relata que o problema persiste?
  → SIM → action = "reset_pppoe"
          params.client_ip = IP que aparece em [Sessão: IP verificado]
          PARE AQUI.
  → NÃO → action = "message"
          message = resposta encerrando ou orientando o cliente

PASSO 5 (somente se reset_pppoe já foi feito nesta conversa E problema ainda persiste):
  → action = "reboot_cpe"
     params.client_ip = IP que aparece em [Sessão: IP verificado]

RETORNE APENAS O JSON. SEM EXPLICAÇÃO. SEM TEXTO ADICIONAL."""

FORMATTER_PROMPT = """Você é ALEX, agente de suporte de internet da empresa.

DADOS DA VERIFICAÇÃO:
{diagnostics}

REGRAS DE OURO PARA O ATENDIMENTO:
1. TRADUZA PARA LEIGOS: O cliente não entende de TI. NUNCA fale "PPPoE", "CPE", "Zabbix", "Ping", "Ansible", "IP" ou "MAC".
   - Troque "Reset de Sessão PPPoE" por "Atualizamos o seu sinal na central".
   - Troque "Reboot do CPE" por "Reiniciamos o seu equipamento remotamente".
2. SEJA CURTO E EMPÁTICO: Responda em no máximo 2 ou 3 frases curtas. 
3. NUNCA DÊ ORDENS TÉCNICAS: Nunca peça para o cliente "reiniciar o roteador da tomada" ou "verificar cabos". Quem resolve o problema somos nós.
4. PERGUNTA FINAL:
   - SE houver problema físico (cabo rompido) ou bloqueio financeiro: Apenas informe o cliente e encerre dizendo que transferiu o atendimento. NÃO pergunte se a internet voltou.
   - SE você executou um reset/reboot lógico: Termine perguntando: "Poderia testar se a conexão voltou a funcionar agora?"
5. ENCERRAMENTO (FALHA): Se os dados indicarem que as tentativas de reset/reboot já foram esgotadas, apenas informe que abriu um chamado e a equipe técnica entrará em contato para agendar uma visita no local."""

# ============================================================
# Utilitários Internos
# ============================================================

def _parse_router_response(text: str) -> dict:
    """Parseia a resposta JSON do decisor com fallback seguro."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    logger.warning("Falha ao parsear JSON do decisor: %r", text[:200])
    return {
        "action": "message",
        "message": "Desculpe, tive uma dificuldade ao processar sua solicitação. Poderia repetir?",
    }


def _extract_ip_from_status(status_text: str) -> str | None:
    """Extrai IP SOMENTE da resposta verificada do Zabbix — nunca do histórico de chat."""
    match = re.search(r"IPs:\s*([\d\.]+)", status_text)
    if match:
        ip = match.group(1)
        return ip if _validate_ip(ip) else None
    return None


def _format_response(llm, diagnostics_str: str, user_message: str = "") -> str:
    """Chama o LLM formatter para traduzir dados técnicos em linguagem amigável."""
    messages = [SystemMessage(content=FORMATTER_PROMPT.format(diagnostics=diagnostics_str))]
    
    # Alguns modelos (especialmente locais via Ollama) falham ou retornam vazio
    # se não houver um HumanMessage na lista de mensagens.
    if user_message:
        messages.append(HumanMessage(content=f"Mensagem do cliente: {user_message}\n\nGere a resposta formatada com base no diagnóstico do sistema acima."))
    else:
        messages.append(HumanMessage(content="Por favor, gere a resposta formatada para o cliente com base no diagnóstico do sistema."))

    try:
        ai_response = llm.invoke(messages)
        return ai_response.content.strip()
    except Exception as exc:
        if GroqRateLimitError and isinstance(exc, GroqRateLimitError):
            logger.warning("Groq RateLimitError no formatter — retornando fallback neutro")
            return (
                "Realizei a verificação do seu equipamento, mas o sistema de atendimento está "
                "sobrecarregado agora e não consigo detalhar o resultado. "
                "Por favor, tente novamente em alguns minutos."
            )
        raise
    except Exception:
        logger.exception("Erro ao formatar resposta de diagnóstico")
        return (
            "Realizei a verificação do seu equipamento, mas ocorreu um problema ao processar "
            "o resultado. Por favor, tente novamente em instantes."
        )


# ============================================================
# Função Principal do Agente
# ============================================================

def run_agent(
    user_message: str,
    conversation_history: list[dict],
    client_ip: str | None = None,
    actions_taken: list[str] | None = None,
) -> tuple[str, list[dict], str | None, list[str]]:
    """
    Decisor de ações por JSON com execução determinística.

    O LLM decide qual ação executar; o Python valida e pode sobrescrever
    para garantir escalada correta (reset → reboot → suporte humano).

    Returns:
        (resposta_final, histórico_atualizado, ip_do_cliente, ações_executadas)
    """
    router_llm = create_router_llm(MAX_TOKENS_ROUTER)
    formatter_llm = create_formatter_llm(MAX_TOKENS_FORMATTER)
    actions_taken = list(actions_taken or [])

    # Filtra histórico: apenas mensagens limpas de usuário/assistente
    clean_history = [
        msg for msg in conversation_history
        if msg.get("role") in ("user", "assistant")
        and msg.get("content", "").strip()
    ]

    # Contexto de sessão injetado no prompt do router
    session_context = ""
    if client_ip and _validate_ip(client_ip):
        session_context += f"\n[Sessão: IP verificado pelo monitoramento = {client_ip}]"
    if actions_taken:
        session_context += f"\n[Sessão: Ações já executadas nesta conversa = {', '.join(actions_taken)}]"

    # Constrói pilha de mensagens — limita histórico para economizar tokens
    messages: list = [SystemMessage(content=ROUTER_PROMPT + session_context)]
    for msg in clean_history[-MAX_HISTORY_TO_LLM:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_message))

    # Chamada ao LLM decisor
    try:
        ai_response = router_llm.invoke(messages)
    except Exception as exc:
        if GroqRateLimitError and isinstance(exc, GroqRateLimitError):
            logger.error("Groq RateLimitError no decisor")
            return (
                "Sistema sobrecarregado no momento. Por favor, tente novamente em alguns instantes.",
                conversation_history,
                client_ip,
                actions_taken,
            )
        logger.exception("Erro ao chamar LLM decisor")
        err_msg = "Problema temporário no sistema. Por favor, tente novamente."
        return err_msg, conversation_history, client_ip, actions_taken

    router_data = _parse_router_response(ai_response.content)
    action = router_data.get("action", "message")
    params = router_data.get("params", {})
    logger.info("Acao decidida pelo LLM: '%s' | params: %s", action, params)

    # -------------------------------------------------------
    # Escalada determinística — Python sobrescreve o LLM
    # para garantir progressão correta independente do modelo
    # -------------------------------------------------------
    if "escalated_financial" in actions_taken:
        logger.info("Sessão travada por bloqueio financeiro")
        return (
            "O seu atendimento já foi transferido para o nosso Setor Financeiro. Por favor, aguarde que um de nossos analistas entrará em contato em breve para te ajudar com a liberação.",
            conversation_history,
            client_ip,
            actions_taken
        )
        
    if "escalated_physical" in actions_taken:
        logger.info("Sessão travada por bloqueio físico")
        return (
            "O seu atendimento já foi encaminhado para a nossa Equipe de Campo devido ao problema físico na fibra. Por favor, aguarde o contato para o agendamento da visita técnica.",
            conversation_history,
            client_ip,
            actions_taken
        )
        
    if "escalated_human" in actions_taken:
        logger.info("Sessão travada por escalonamento técnico final")
        return (
            "O seu chamado já está na fila da nossa equipe técnica de campo e eles entrarão em contato em breve. Agradecemos a compreensão!",
            conversation_history,
            client_ip,
            actions_taken
        )
    if "reboot_cpe" in actions_taken and action in ("reset_pppoe", "reboot_cpe"):
        logger.info("reboot_cpe já executado → escalando para suporte humano")
        action = "human_escalation"
    elif "reset_pppoe" in actions_taken and action == "reset_pppoe":
        logger.info("reset_pppoe já executado → escalando para reboot_cpe")
        action = "reboot_cpe"

    logger.info("Acao final: '%s'", action)

    final_response = ""
    new_client_ip = client_ip

    # -------------------------------------------------------
    # Despacho de Ações — determinístico
    # -------------------------------------------------------

    if action in ("message", "ask_identification"):
        # Resposta direta do LLM — sem chamada de ferramenta
        final_response = router_data.get("message", "Como posso ajudar?")

    elif action == "diagnose":
        identifier = _clean_identifier((params.get("identifier") or "").strip())

        if not _validate_identifier(identifier):
            final_response = (
                "Por favor, informe seu nome completo ou número do contrato para prosseguir."
            )
        else:
            logger.info("Iniciando diagnóstico para identificador: '%s'", identifier)
            status_res = ""
            incidents_res = ""
            router_diag = ""
            new_client_ip = None

            # Passo 1: Status no Zabbix
            try:
                status_res = get_client_status.invoke({"client_id_or_mac": identifier})
                new_client_ip = _extract_ip_from_status(status_res)
            except Exception:
                logger.exception("Erro em get_client_status para '%s'", identifier)
                status_res = "Não foi possível consultar o monitoramento."

            # Passos 2 e 3: apenas se o equipamento foi localizado
            if new_client_ip:
                try:
                    incidents_res = get_active_incidents.invoke({"client_id_or_mac": identifier})
                except Exception:
                    logger.exception("Erro em get_active_incidents")
                    incidents_res = "Não foi possível verificar alertas ativos."

                # ---- TRIAGEM RÁPIDA (Ignora Ansible se o problema for físico ou financeiro) ----
                is_financial = incidents_res and "Bloqueio Financeiro" in incidents_res
                is_physical = incidents_res and ("Fibra Rompida" in incidents_res or "Sinal" in incidents_res)

                if is_financial:
                    logger.info("Triagem: Bloqueio financeiro detectado.")
                    if "escalated_financial" not in actions_taken:
                        actions_taken.append("escalated_financial")
                    router_diag = (
                        "INSTRUÇÃO DE SISTEMA: O cliente está bloqueado por falta de pagamento. "
                        "Nenhuma correção técnica pode ser feita. Informe ao cliente, com muita educação, "
                        "sobre a pendência financeira e diga que você está transferindo o atendimento para o "
                        "Setor Financeiro resolver a situação."
                    )
                elif is_physical:
                    logger.info("Triagem: Problema físico detectado na fibra.")
                    if "escalated_physical" not in actions_taken:
                        actions_taken.append("escalated_physical")
                    router_diag = (
                        "INSTRUÇÃO DE SISTEMA: O sinal óptico do cliente caiu, a fibra está rompida. "
                        "Avise o cliente que identificamos um problema físico no cabo que chega "
                        "à residência e que nossa equipe técnica de campo será enviada para o conserto. "
                        "NÃO mande reiniciar o equipamento."
                    )
                else:
                    # Problema lógico ou sem alertas — Executa diagnóstico Ansible (pode demorar ~10s)
                    try:
                        logger.info("Executando diagnóstico SSH no IP: %s", new_client_ip)
                        router_diag = run_router_diagnostic.invoke({"client_ip": new_client_ip})
                    except Exception:
                        logger.exception("Erro em run_router_diagnostic para IP '%s'", new_client_ip)
                        router_diag = "Não foi possível conectar ao equipamento remotamente."

            # Ação autônoma: se há incidente ativo, executa reset imediatamente sem pedir ao cliente
            # Ação autônoma: só executa se há incidente E NÃO for financeiro/físico
            auto_action_res = ""
            if new_client_ip and _has_active_incidents(incidents_res) and not (is_financial or is_physical):
                if "reset_pppoe" not in actions_taken:
                    try:
                        logger.info("Incidente lógico detectado — executando reset PPPoE para %s", new_client_ip)
                        auto_action_res = reset_pppoe_connection.invoke({"client_ip": new_client_ip})
                        actions_taken.append("reset_pppoe")
                    except Exception:
                        logger.exception("Erro no reset automático para '%s'", new_client_ip)
                        auto_action_res = "Não foi possível executar o reset automático."
                elif "reboot_cpe" not in actions_taken:
                    try:
                        logger.info("reset_pppoe já feito — executando reboot automaticamente para %s", new_client_ip)
                        auto_action_res = reboot_client_cpe.invoke({"client_ip": new_client_ip})
                        actions_taken.append("reboot_cpe")
                    except Exception:
                        logger.exception("Erro no reboot automático para '%s'", new_client_ip)
                        auto_action_res = "Não foi possível executar o reboot automático."
                else:
                    auto_action_res = "Reset e reboot já foram executados anteriormente sem sucesso."
            
            # Ajuste da mensagem final da ação para o LLM não se confundir
            if is_financial:
                final_action_msg = "Nenhuma ação técnica possível. Motivo: Bloqueio financeiro."
            elif is_physical:
                final_action_msg = "Nenhuma ação técnica possível. Motivo: Problema físico na fibra."
            else:
                final_action_msg = auto_action_res or "Nenhuma — conexão estável"

            diag_summary = (
                f"Status do monitoramento: {status_res}\n"
                f"Alertas ativos: {incidents_res or 'nenhum verificado'}\n"
                f"Diagnóstico do equipamento: {router_diag or 'equipamento não localizado na rede'}\n"
                f"Ação corretiva executada automaticamente: {final_action_msg}"
            )
            logger.info("Diagnóstico consolidado:\n%s", diag_summary)
            final_response = _format_response(formatter_llm, diag_summary, user_message)

    elif action in ("reset_pppoe", "reboot_cpe"):
        # Segurança: IP SOMENTE do estado verificado da sessão — nunca de params do LLM ou do chat
        # Isso impede que o usuário manipule qual equipamento será afetado via injeção de prompt
        if not client_ip or not _validate_ip(client_ip):
            final_response = (
                "Para executar essa ação, preciso localizar seu equipamento primeiro. "
                "Pode me informar seu nome completo ou número do contrato?"
            )
        else:
            try:
                if action == "reset_pppoe":
                    logger.info("Executando reset PPPoE para IP: %s", client_ip)
                    action_res = reset_pppoe_connection.invoke({"client_ip": client_ip})
                    actions_taken.append("reset_pppoe")
                else:
                    logger.info("Executando reboot CPE para IP: %s", client_ip)
                    action_res = reboot_client_cpe.invoke({"client_ip": client_ip})
                    actions_taken.append("reboot_cpe")
            except Exception:
                logger.exception("Erro ao executar acao '%s' para IP '%s'", action, client_ip)
                action_res = "Não foi possível executar a ação no equipamento no momento."

            exec_summary = f"Acao: {action}\nAlvo: {client_ip}\nResultado: {action_res}"
            final_response = _format_response(formatter_llm, exec_summary, user_message)

    elif action == "human_escalation":
        logger.info("Escalando para suporte humano após reset e reboot sem sucesso")
        if "escalated_human" not in actions_taken:
            actions_taken.append("escalated_human")
        final_response = (
            "Já realizei todas as ações remotas disponíveis e o problema infelizmente persiste. "
            "Vou acionar nossa equipe técnica para uma visita presencial. "
            "Em breve um técnico entrará em contato com você para solucionar a questão."
        )

    else:
        # Ação desconhecida — fallback seguro
        logger.warning("Acao desconhecida recebida do decisor: '%s'", action)
        final_response = router_data.get("message", "Como posso ajudar?")

    # Atualiza histórico com mensagens limpas
    updated_history = conversation_history.copy()
    updated_history.append({"role": "user", "content": user_message})
    updated_history.append({"role": "assistant", "content": final_response})

    return final_response, updated_history, new_client_ip or client_ip, actions_taken
