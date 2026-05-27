# ============================================================
# tools/zabbix.py
# Ferramentas de LEITURA do Zabbix para o Agente NOC
#
# IMPORTANTE: Este módulo é SOMENTE LEITURA. Nenhuma ação
# destrutiva ou de modificação é executada aqui.
# A autenticação usa token de API (Zabbix 5.4+).
# ============================================================

import os
import logging
import requests

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Configuração da Conexão Zabbix ---
ZABBIX_URL = os.getenv("ZABBIX_URL", "http://zabbix.sua-empresa.com/api_jsonrpc.php")
ZABBIX_TOKEN = os.getenv("ZABBIX_TOKEN", "")  # Token de API (Zabbix 5.4+)
ZABBIX_TIMEOUT = int(os.getenv("ZABBIX_TIMEOUT", "10"))  # segundos


# ============================================================
# Utilitário Interno: Chamada à API do Zabbix
# ============================================================

@retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def _zabbix_request(method: str, params: dict) -> dict:
    """
    Realiza uma chamada JSON-RPC à API do Zabbix.
    
    Implementa retry automático com backoff exponencial em caso de
    timeout ou falha de conexão (até 2 tentativas, 1-4s de espera).

    Args:
        method: Método da API Zabbix (ex: 'host.get', 'trigger.get').
        params: Parâmetros do método em formato de dicionário.

    Returns:
        O campo 'result' da resposta JSON da API.

    Raises:
        ConnectionError: Se a API do Zabbix estiver inacessível após retries.
        PermissionError: Se o token de API for inválido ou expirado.
        ValueError: Se a API retornar um erro de aplicação.
    """
    if not ZABBIX_TOKEN:
        raise PermissionError(
            "ZABBIX_TOKEN não configurado. Verifique o arquivo .env"
        )

    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZABBIX_TOKEN}",
    }

    try:
        response = requests.post(
            ZABBIX_URL,
            json=payload,
            headers=headers,
            timeout=ZABBIX_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            error_data = data["error"]
            code = error_data.get("code", -1)
            message = error_data.get("message", "Erro desconhecido")
            data_msg = error_data.get("data", "")

            if code in (-32602, -32500):  # Códigos de autenticação/permissão
                raise PermissionError(f"Zabbix API Auth Error: {message} - {data_msg}")
            raise ValueError(f"Zabbix API Error [{code}]: {message} - {data_msg}")

        return data.get("result", {})

    except requests.Timeout:
        logger.error("Timeout na API do Zabbix após %ds", ZABBIX_TIMEOUT)
        raise
    except requests.ConnectionError as e:
        logger.error("Falha de conexão com Zabbix: %s", e)
        raise


def _find_host(client_identifier: str) -> dict | list | None:
    """
    Busca um host no Zabbix por multiplos criterios, com prioridade em Tags.

    Estrategia em cascata:
      1. Tags EXATAS: testa 'cliente','Cliente','CLIENTE','contrato','Contrato', etc.
      2. Tags PARCIAIS (contains): mesmas variantes de capitalizacao.
      3. Varredura client-side: busca o valor em QUALQUER tag, case-insensitive.
      4. Nome do host (campo host/name).
      5. IP da interface de monitoramento.

    Returns:
        dict  -> 1 host encontrado
        list  -> multiplos hosts (ambiguidade)
        None  -> nao encontrado
    """
    TAG_KEYS = ["cliente", "contrato", "cpf", "telefone"]

    BASE_PARAMS = {
        "output": ["hostid", "host", "name", "status", "description"],
        "selectInterfaces": ["ip", "type", "useip", "dns"],
        "selectTags": "extend",
        "limit": 10,
    }

    try:
        # 1. Busca EXATA por Tags (todas as capitalizacoes)
        for tag_key in TAG_KEYS:
            for variant in [tag_key, tag_key.capitalize(), tag_key.upper()]:
                result = _zabbix_request("host.get", {
                    **BASE_PARAMS,
                    "tags": [{
                        "tag": variant,
                        "value": client_identifier,
                        "operator": "1",
                    }],
                })
                if result:
                    logger.debug("Host por tag exata [%s]=[%s]: %s",
                                 variant, client_identifier, result[0].get("name"))
                    return result[0]

        # 2. Busca PARCIAL por Tags (contains, todas as capitalizacoes)
        for tag_key in TAG_KEYS:
            for variant in [tag_key, tag_key.capitalize(), tag_key.upper()]:
                result = _zabbix_request("host.get", {
                    **BASE_PARAMS,
                    "tags": [{
                        "tag": variant,
                        "value": client_identifier,
                        "operator": "0",
                    }],
                })
                if result:
                    if len(result) == 1:
                        logger.debug("Host por tag parcial [%s]: %s",
                                     variant, result[0].get("name"))
                        return result[0]
                    else:
                        logger.info("Ambiguidade: %d hosts para [%s]",
                                    len(result), client_identifier)
                        return result

        # 3. Varredura client-side: qualquer chave de tag, case-insensitive
        all_hosts = _zabbix_request("host.get", {**BASE_PARAMS, "limit": 500})
        id_lower = client_identifier.lower()
        matched = [
            h for h in all_hosts
            if any(id_lower == t.get("value", "").lower() for t in h.get("tags", []))
        ]
        if matched:
            if len(matched) == 1:
                logger.debug("Host por varredura de tags: %s", matched[0].get("name"))
                return matched[0]
            return matched

        # 4. Nome do host
        result = _zabbix_request("host.get", {
            **BASE_PARAMS,
            "search": {"host": client_identifier, "name": client_identifier},
            "searchByAny": True,
            "searchWildcardsEnabled": True,
        })
        if result:
            return result[0] if len(result) == 1 else result

        # 5. IP da interface
        result = _zabbix_request("host.get", {
            **BASE_PARAMS,
            "filter": {"ip": client_identifier},
        })
        if result:
            return result[0]

        logger.info("Nenhum host encontrado para: [%s]", client_identifier)
        return None

    except Exception as e:
        logger.warning("Falha ao buscar host [%s]: %s", client_identifier, e)
        return None


# ============================================================
# TOOL 1: Status do Cliente
# ============================================================

@tool
def get_client_status(client_id_or_mac: str) -> str:
    """Primeiro passo: verifica se o equipamento do cliente esta no monitoramento. Busca por nome, tag (cliente/contrato), IP ou hostname. Args: client_id_or_mac - nome do cliente, numero do contrato, IP ou hostname."""
    try:
        host = _find_host(client_id_or_mac)

        if not host:
            return (
                f"⚠️ Nenhum equipamento encontrado no Zabbix com o identificador "
                f"'{client_id_or_mac}'. Verifique se o host está cadastrado no sistema "
                f"de monitoramento ou tente com outro identificador (IP, MAC, hostname)."
            )

        # Status 0 = Ativo (monitorado), Status 1 = Desabilitado
        status_map = {
            "0": "🟢 ATIVO (sendo monitorado)",
            "1": "🔴 DESABILITADO (monitoramento suspenso)",
        }
        status_label = status_map.get(str(host.get("status", "-1")), "❓ DESCONHECIDO")

        # Extrai IPs das interfaces
        interfaces = host.get("interfaces", [])
        ips = [iface.get("ip", "N/A") for iface in interfaces if iface.get("ip")]
        ip_str = ", ".join(ips) if ips else "Não disponível"

        return (
            f"📡 Host encontrado no Zabbix:\n"
            f"  - Nome: {host.get('name', host.get('host', 'N/A'))}\n"
            f"  - Host ID: {host.get('hostid', 'N/A')}\n"
            f"  - IPs: {ip_str}\n"
            f"  - Status de Monitoramento: {status_label}"
        )

    except PermissionError as e:
        return f"🔐 Erro de autenticação no Zabbix: {e}. Contate o administrador do sistema."
    except (requests.Timeout, requests.ConnectionError):
        return (
            "🔌 Sistema de diagnóstico indisponível: Não foi possível conectar ao "
            "Zabbix. O servidor pode estar sobrecarregado ou inacessível no momento. "
            "Tente novamente em alguns instantes."
        )
    except Exception:
        logger.exception("Erro inesperado em get_client_status para '%s'", client_id_or_mac)
        return "Erro interno ao consultar o monitoramento. Tente novamente."


# ============================================================
# TOOL 2: Incidentes Ativos
# ============================================================

@tool
def get_active_incidents(client_id_or_mac: str) -> str:
    """Busca alertas e problemas ativos do cliente no monitoramento. Usar apos get_client_status. Args: client_id_or_mac - mesmo identificador do get_client_status."""
    try:
        host = _find_host(client_id_or_mac)

        if not host:
            return (
                f"⚠️ Host '{client_id_or_mac}' não encontrado no Zabbix. "
                f"Não é possível buscar incidentes de um host não monitorado."
            )

        host_id = host["hostid"]

        # Busca triggers ativas (value=1 = PROBLEM, only_true=1 = somente disparadas)
        triggers = _zabbix_request("trigger.get", {
            "output": ["triggerid", "description", "priority", "lastchange", "comments"],
            "hostids": host_id,
            "filter": {"value": 1},          # 1 = PROBLEM (ativo)
            "only_true": 1,                   # Apenas triggers que já foram disparadas
            "skipDependent": 1,               # Ignora triggers dependentes de outras
            "sortfield": "priority",
            "sortorder": "DESC",             # Críticos primeiro
            "limit": 20,
        })

        if not triggers:
            return (
                f"✅ Nenhum incidente ativo encontrado para o host "
                f"'{host.get('name', client_id_or_mac)}'. "
                f"O monitoramento não detectou problemas no momento."
            )

        # Mapa de severidade (prioridade no Zabbix)
        severity_map = {
            "0": "ℹ️ Não classificado",
            "1": "📘 Informação",
            "2": "⚠️ Aviso",
            "3": "🟠 Média",
            "4": "🔴 Alta",
            "5": "🚨 Desastre",
        }

        lines = [
            f"🚨 {len(triggers)} incidente(s) ativo(s) para '{host.get('name', client_id_or_mac)}':\n"
        ]

        for i, trigger in enumerate(triggers, 1):
            severity = severity_map.get(str(trigger.get("priority", "0")), "❓")
            description = trigger.get("description", "Sem descrição")
            comments = trigger.get("comments", "")

            # Converte timestamp Unix para data legível
            last_change_ts = trigger.get("lastchange", 0)
            if last_change_ts:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(int(last_change_ts), tz=timezone.utc)
                last_change_str = dt.strftime("%d/%m/%Y %H:%M:%S UTC")
            else:
                last_change_str = "Desconhecido"

            lines.append(
                f"  [{i}] {severity}\n"
                f"      Problema: {description}\n"
                f"      Desde: {last_change_str}\n"
                + (f"      Detalhe: {comments}\n" if comments else "")
            )

        return "\n".join(lines)

    except PermissionError as e:
        return f"🔐 Erro de autenticação no Zabbix: {e}."
    except (requests.Timeout, requests.ConnectionError):
        return (
            "🔌 Sistema de diagnóstico indisponível: Timeout ao conectar no Zabbix. "
            "Tente novamente em instantes."
        )
    except Exception:
        logger.exception("Erro inesperado em get_active_incidents para '%s'", client_id_or_mac)
        return "Erro interno ao buscar alertas. Tente novamente."


# ============================================================
# TOOL 3: Sinal Óptico
# ============================================================

@tool
def check_optical_signal(client_id_or_mac: str) -> str:
    """Verifica qualidade do sinal do cabo de fibra do cliente. Usar quando cliente relata lentidao ou quedas frequentes. Args: client_id_or_mac - identificador do cliente."""
    try:
        host = _find_host(client_id_or_mac)

        if not host:
            return (
                f"⚠️ Host '{client_id_or_mac}' não encontrado no Zabbix. "
                f"Impossível verificar sinal óptico."
            )

        host_id = host["hostid"]
        host_name = host.get("name", client_id_or_mac)

        # Busca itens relacionados a sinal óptico por palavras-chave comuns
        optical_keywords = ["optical", "rx_power", "tx_power", "pon", "dbm", "signal", "atenuacao", "sinal"]
        
        items = _zabbix_request("item.get", {
            "output": ["itemid", "name", "lastvalue", "lastclock", "units", "key_"],
            "hostids": host_id,
            "search": {
                "key_": "optical",
                "name": "optical"
            },
            "searchByAny": True,
            "limit": 20,
        })

        # Fallback: busca por "pon" se não encontrou "optical"
        if not items:
            items = _zabbix_request("item.get", {
                "output": ["itemid", "name", "lastvalue", "lastclock", "units", "key_"],
                "hostids": host_id,
                "search": {"name": "power"},
                "limit": 20,
            })

        if not items:
            return (
                f"📡 Nenhum item de sinal óptico encontrado para o host '{host_name}'. "
                f"Possíveis causas:\n"
                f"  1. Este cliente não possui ONU/ONT monitorada (pode ser rede metálica/rádio)\n"
                f"  2. Os itens de monitoramento óptico não estão configurados no template Zabbix\n"
                f"  3. O host está usando chaves de monitoramento não padrão\n"
                f"\nNeste caso, avalie outros indicadores (incidentes, ping, latência)."
            )

        # Analisa os valores coletados
        lines = [f"📡 Sinal Óptico para '{host_name}':\n"]

        for item in items:
            item_name = item.get("name", "Item sem nome")
            last_value = item.get("lastvalue", "N/A")
            units = item.get("units", "")
            last_clock = item.get("lastclock", 0)

            # Formata timestamp
            if last_clock:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(int(last_clock), tz=timezone.utc)
                clock_str = dt.strftime("%d/%m/%Y %H:%M:%S UTC")
            else:
                clock_str = "Sem dados recentes"

            # Analisa qualidade do sinal para itens em dBm (RX Power)
            quality_label = ""
            try:
                value_float = float(last_value)
                if "rx" in item_name.lower() or "receive" in item_name.lower() or "dbm" in units.lower():
                    if value_float >= -27:
                        quality_label = " ✅ ÓTIMO"
                    elif value_float >= -30:
                        quality_label = " ⚠️ FRACO - ATENÇÃO"
                    else:
                        quality_label = " 🚨 CRÍTICO - SERVIÇO EM RISCO"
            except (ValueError, TypeError):
                pass

            lines.append(
                f"  🔹 {item_name}\n"
                f"     Valor: {last_value} {units}{quality_label}\n"
                f"     Última leitura: {clock_str}\n"
            )

        return "\n".join(lines)

    except PermissionError as e:
        return f"🔐 Erro de autenticação no Zabbix: {e}."
    except (requests.Timeout, requests.ConnectionError):
        return (
            "🔌 Sistema de diagnóstico indisponível: Não foi possível conectar ao Zabbix. "
            "Tente novamente em instantes."
        )
    except Exception:
        logger.exception("Erro inesperado em check_optical_signal para '%s'", client_id_or_mac)
        return "Erro interno ao verificar sinal. Tente novamente."
