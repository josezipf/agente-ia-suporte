# ============================================================
# tools/ansible.py
# Ferramentas de EXECUÇÃO do Agente NOC via Ansible
#
# IMPORTANTE: Este módulo executa AÇÕES REAIS nos equipamentos.
# Use SOMENTE após diagnóstico via ferramentas do Zabbix.
# Cada playbook é idempotente e seguro para reexecução.
# ============================================================

import os
import re
import sys
import logging
import ansible_runner

from langchain_core.tools import tool
from dotenv import load_dotenv

_VALID_IP_RE = re.compile(r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$')


def _validate_ip(value: str) -> bool:
    """Verifica se o IP é um IPv4 válido antes de usar em comandos."""
    if not value or not _VALID_IP_RE.match(value):
        return False
    return all(0 <= int(p) <= 255 for p in value.split('.'))

load_dotenv()

# Garante que a pasta bin do venv atual esteja no PATH.
# Usamos a localização relativa ao arquivo para garantir resolução absoluta do venv do projeto,
# ignorando qualquer variação causada pelo reloader do Uvicorn ou sys.prefix.
venv_bin_relative = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "venv", "bin")
venv_bin_exec = os.path.dirname(sys.executable)

logger = logging.getLogger(__name__)

for bin_dir in (venv_bin_relative, venv_bin_exec):
    if os.path.isdir(bin_dir) and bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.path.pathsep + os.environ.get("PATH", "")
        logger.info("Injetado diretório bin do venv no PATH: %s", bin_dir)

logger.info("PATH atual para execução do Ansible: %s", os.environ.get("PATH"))

# --- Configuração do Ansible ---
_playbooks_dir_env = os.getenv("ANSIBLE_PLAYBOOKS_DIR", "")
if not _playbooks_dir_env or _playbooks_dir_env == "./playbooks":
    PLAYBOOKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "playbooks")
else:
    PLAYBOOKS_DIR = os.path.abspath(_playbooks_dir_env)

ANSIBLE_INVENTORY = os.getenv("ANSIBLE_INVENTORY", "/etc/ansible/hosts")
ANSIBLE_TIMEOUT = int(os.getenv("ANSIBLE_TIMEOUT", "120"))  # segundos por playbook
ANSIBLE_VERBOSITY = int(os.getenv("ANSIBLE_VERBOSITY", "1"))  # 0=silencioso, 4=debug
ANSIBLE_USER = os.getenv("ANSIBLE_USER", "admin")


# ============================================================
# Utilitário Interno: Execução de Playbook
# ============================================================

def _run_playbook(playbook_name: str, extra_vars: dict) -> dict:
    """
    Executa um playbook Ansible de forma síncrona usando ansible-runner.

    Gerencia o ciclo de vida completo da execução: criação do diretório
    de artefatos temporário, execução, coleta de logs e limpeza.

    Args:
        playbook_name: Nome do arquivo do playbook (ex: 'router_diagnostic.yml').
        extra_vars: Variáveis extras passadas ao playbook (ex: {'target_ip': '...'}).

    Returns:
        Dicionário com:
          - 'success' (bool): True se rc == 0 (sem erros)
          - 'rc' (int): Return code do Ansible (0=OK, outros=falha)
          - 'status' (str): Status do runner ('successful', 'failed', 'timeout', etc.)
          - 'stdout' (str): Saída completa da execução
          - 'errors' (list): Lista de mensagens de erro capturadas

    Raises:
        FileNotFoundError: Se o playbook especificado não existir.
        TimeoutError: Se a execução ultrapassar ANSIBLE_TIMEOUT segundos.
    """
    playbook_path = os.path.join(PLAYBOOKS_DIR, playbook_name)

    if not os.path.isfile(playbook_path):
        raise FileNotFoundError(
            f"Playbook '{playbook_name}' não encontrado em '{PLAYBOOKS_DIR}'. "
            f"Crie o playbook ou verifique o caminho ANSIBLE_PLAYBOOKS_DIR no .env"
        )

    # Injeta automaticamente o usuário configurado do Ansible
    if "ansible_user" not in extra_vars:
        extra_vars["ansible_user"] = ANSIBLE_USER

    # Injeta desativação de TTY para evitar travamento em RouterOS
    if "ansible_ssh_use_tty" not in extra_vars:
        extra_vars["ansible_ssh_use_tty"] = False

    target_host = extra_vars.get("target_host")
    if target_host:
        inventory_param = {
            "all": {
                "hosts": {
                    target_host: {
                        "ansible_user": extra_vars.get("ansible_user", ANSIBLE_USER),
                        "ansible_ssh_use_tty": False
                    }
                }
            }
        }
    else:
        inventory_param = ANSIBLE_INVENTORY

    logger.info("Iniciando playbook '%s' com vars: %s (inventário: %s)", playbook_name, extra_vars, target_host or inventory_param)

    try:
        runner_result = ansible_runner.run(
            playbook=playbook_path,
            inventory=inventory_param,
            extravars=extra_vars,
            verbosity=ANSIBLE_VERBOSITY,
            timeout=ANSIBLE_TIMEOUT,
            quiet=False,
        )

        # Coleta eventos de erro e outputs úteis
        errors = []
        outputs = {}
        for event in runner_result.events:
            ev_type = event.get("event")
            event_data = event.get("event_data", {})
            task = event_data.get("task", "Tarefa desconhecida")
            res = event_data.get("res", {})
            
            if ev_type in ("runner_on_failed", "runner_on_unreachable"):
                msg = res.get("msg", str(res))
                errors.append(f"[{task}] {msg}")
            elif ev_type == "runner_on_ok":
                # Captura outputs de tarefas de debug (msg) ou comandos (stdout_lines)
                if "stdout_lines" in res:
                    outputs[task] = "\n".join(res["stdout_lines"])
                elif "msg" in res:
                    outputs[task] = res["msg"]

        success = runner_result.rc == 0 and runner_result.status == "successful"

        return {
            "success": success,
            "rc": runner_result.rc,
            "status": runner_result.status,
            "outputs": outputs,
            "errors": errors,
        }

    except TimeoutError:
        logger.error("Timeout após %ds executando '%s'", ANSIBLE_TIMEOUT, playbook_name)
        raise
    except Exception as e:
        logger.exception("Erro inesperado ao executar playbook '%s'", playbook_name)
        raise RuntimeError(f"Falha ao executar Ansible: {type(e).__name__}: {e}") from e


import json
import re

def _format_result(result: dict, action_name: str, target: str) -> str:
    """
    Formata o resultado de uma execução Ansible para um JSON ultrassimplificado.
    Converte tabelas brutas do RouterOS em status mastigados (OK/Falha) para o LLM.
    """
    summary = {
        "acao": action_name,
        "alvo": target,
        "sucesso": result["success"],
    }
    
    if result["success"]:
        parsed_data = {}
        for task, output in result.get("outputs", {}).items():
            if not output: continue
            
            output_lower = output.lower()
            task_lower = task.lower()
            
            # 1. Parsing do Ping
            if "ping" in task_lower:
                loss_match = re.search(r"packet-loss=(\d+)%", output_lower)
                if loss_match:
                    loss = int(loss_match.group(1))
                    parsed_data["internet"] = "Offline (100% perda de pacotes)" if loss == 100 else f"Online (perda {loss}%)"
                elif "timeout" in output_lower:
                    parsed_data["internet"] = "Offline (Timeouts)"

            # 2. Parsing do PPPoE
            elif "pppoe" in task_lower or "pppoe" in output_lower:
                if "status: disconnected" in output_lower:
                    parsed_data["conexao_pppoe"] = "Problema - Desconectado"
                elif "status: connected" in output_lower:
                    parsed_data["conexao_pppoe"] = "OK - Conectado"

            # 3. Parsing de Recursos (CPU e Uptime)
            elif "recursos" in task_lower or "uptime" in output_lower:
                uptime_match = re.search(r"uptime:\s*([^\n\r]+)", output)
                cpu_match = re.search(r"cpu-load:\s*([^\n\r]+)", output)
                if uptime_match:
                    parsed_data["roteador_ligado_ha"] = uptime_match.group(1).strip()
                if cpu_match:
                    parsed_data["uso_cpu"] = cpu_match.group(1).strip()

        summary["diagnostico"] = parsed_data if parsed_data else "Comando executado com sucesso."
            
    else:
        summary["erros"] = result.get("errors", [])
        summary["recomendacao"] = "Equipamento inacessível ou erro na execução."
        
    return json.dumps(summary, ensure_ascii=False, indent=2)


# ============================================================
# TOOL 4: Diagnóstico do Roteador
# ============================================================

@tool
def run_router_diagnostic(client_ip: str) -> str:
    """Diagnostico remoto no equipamento do cliente via SSH. Nao causa interrupcao. Usar antes de acoes corretivas. Args: client_ip - IP do equipamento na rede de gerencia."""
    if not _validate_ip(client_ip):
        logger.error("IP inválido recebido em run_router_diagnostic: %r", client_ip)
        return "Erro: IP do equipamento inválido."
    try:
        logger.info("Executando diagnóstico no roteador do cliente IP=%s", client_ip)
        result = _run_playbook(
            playbook_name="router_diagnostic.yml",
            extra_vars={"target_host": client_ip, "target_ip": client_ip},
        )
        return _format_result(result, "Diagnóstico do Roteador", client_ip)

    except FileNotFoundError:
        return "Playbook de diagnóstico não encontrado. Contate o administrador."
    except TimeoutError:
        return f"Timeout: O equipamento {client_ip} não respondeu em {ANSIBLE_TIMEOUT}s."
    except RuntimeError as e:
        logger.error("RuntimeError em run_router_diagnostic IP=%s: %s", client_ip, e)
        return "Falha na execução do diagnóstico remoto."
    except Exception:
        logger.exception("Erro inesperado em run_router_diagnostic para IP '%s'", client_ip)
        return "Erro interno ao executar diagnóstico."


# ============================================================
# TOOL 5: Reset de Conexão PPPoE
# ============================================================

@tool
def reset_pppoe_connection(client_ip: str) -> str:
    """Reinicia a conexao do cliente (~30s de interrupcao). Usar para conexoes travadas. Args: client_ip - IP do equipamento."""
    if not _validate_ip(client_ip):
        logger.error("IP inválido recebido em reset_pppoe_connection: %r", client_ip)
        return "Erro: IP do equipamento inválido."
    try:
        logger.info("Executando reset PPPoE para cliente IP=%s", client_ip)
        result = _run_playbook(
            playbook_name="reset_pppoe.yml",
            extra_vars={"target_host": client_ip, "client_ip": client_ip},
        )
        base_response = _format_result(result, "Reset de Sessão PPPoE", client_ip)
        if result["success"]:
            base_response += "\nAguarde 30 segundos para a sessão reestabelecer."
        return base_response

    except FileNotFoundError:
        return "Playbook de reset não encontrado. Contate o administrador."
    except TimeoutError:
        return f"Timeout: Reset PPPoE para {client_ip} não concluiu em {ANSIBLE_TIMEOUT}s."
    except RuntimeError as e:
        logger.error("RuntimeError em reset_pppoe_connection IP=%s: %s", client_ip, e)
        return "Falha na execução do reset de conexão."
    except Exception:
        logger.exception("Erro inesperado em reset_pppoe_connection para IP '%s'", client_ip)
        return "Erro interno ao executar reset."


# ============================================================
# TOOL 6: Reboot do CPE do Cliente
# ============================================================

@tool
def reboot_client_cpe(client_ip: str) -> str:
    """ULTIMO RECURSO: reinicia o roteador do cliente remotamente (3-5min de interrupcao). Args: client_ip - IP do equipamento."""
    if not _validate_ip(client_ip):
        logger.error("IP inválido recebido em reboot_client_cpe: %r", client_ip)
        return "Erro: IP do equipamento inválido."
    try:
        logger.info("Executando reboot do CPE do cliente IP=%s", client_ip)
        result = _run_playbook(
            playbook_name="reboot_cpe.yml",
            extra_vars={"target_host": client_ip, "cpe_ip": client_ip},
        )
        base_response = _format_result(result, "Reboot do CPE", client_ip)
        if result["success"]:
            base_response += "\nAguarde 3-5 minutos para o equipamento reiniciar completamente."
        else:
            base_response += "\nReboot remoto falhou. Pode ser necessário reiniciar o equipamento manualmente."
        return base_response

    except FileNotFoundError:
        return "Playbook de reboot não encontrado. Contate o administrador."
    except TimeoutError:
        return f"Timeout: Reboot do CPE {client_ip} não concluiu em {ANSIBLE_TIMEOUT}s. Verifique o equipamento em alguns minutos."
    except RuntimeError as e:
        logger.error("RuntimeError em reboot_client_cpe IP=%s: %s", client_ip, e)
        return "Falha na execução do reboot remoto."
    except Exception:
        logger.exception("Erro inesperado em reboot_client_cpe para IP '%s'", client_ip)
        return "Erro interno ao executar reboot."
