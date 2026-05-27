# 🤖 ALEX - Agente de Suporte NOC para ISP

Agente de IA de Nível 1/2 para atendimento ao cliente de Provedores de Internet.  
Diagnostica problemas de rede autonomamente via Zabbix e executa manutenções via Ansible.

## 🏗️ Arquitetura

```
Cliente (WhatsApp/Portal) → POST /chat (FastAPI)
                                    ↓
                            agent.py (Loop ReAct)
                           /                    \
                  tools/zabbix.py          tools/ansible.py
                  (LEITURA)                (EXECUÇÃO)
                       ↓                        ↓
                  API Zabbix              Playbooks Ansible
                  (Diagnóstico)           (Manutenção)
```

## 📁 Estrutura de Arquivos

```
AGENTE SUPORTE/
├── main.py                   # FastAPI: rota /chat + sessões
├── agent.py                  # LLM Groq + System Prompt + Loop ReAct
├── tools/
│   ├── zabbix.py             # 3 ferramentas de leitura (Zabbix)
│   └── ansible.py            # 3 ferramentas de execução (Ansible)
├── playbooks/
│   ├── router_diagnostic.yml # Coleta métricas sem alterar config
│   ├── reset_pppoe.yml       # Reinicia sessão PPPoE (~15s down)
│   └── reboot_cpe.yml        # Reboot completo do CPE (~5min down)
├── .env.example              # Template de variáveis de ambiente
├── .gitignore
└── requirements.txt
```

## ⚡ Instalação Rápida

```bash
# 1. Clone e entre no diretório
cd "AGENTE SUPORTE"

# 2. Crie e ative o ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure as variáveis de ambiente
cp .env.example .env
nano .env  # Preencha GROQ_API_KEY, ZABBIX_URL, ZABBIX_TOKEN

# 5. Inicie o servidor
python main.py
# Ou com uvicorn diretamente:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🔧 Configuração (.env)

| Variável | Descrição | Exemplo |
|----------|-----------|---------|
| `GROQ_API_KEY` | Chave da API Groq | `gsk_xxx...` |
| `GROQ_MODEL` | Modelo LLM | `llama3-70b-8192` |
| `ZABBIX_URL` | URL da API Zabbix | `http://zabbix.local/api_jsonrpc.php` |
| `ZABBIX_TOKEN` | Token de API Zabbix 5.4+ | `abc123...` |
| `ANSIBLE_PLAYBOOKS_DIR` | Caminho dos playbooks | `./playbooks` |
| `ANSIBLE_INVENTORY` | Inventário Ansible | `/etc/ansible/hosts` |

## 🌐 API Endpoints

### `POST /chat` — Chat com o Agente
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Minha internet está caída, sou o cliente joao-silva-fibra",
    "session_id": null
  }'
```

**Resposta:**
```json
{
  "response": "Olá! Sou o ALEX. Vou verificar o status do seu equipamento agora...",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message_count": 2,
  "timestamp": "2025-01-15T14:30:00Z"
}
```

### `GET /health` — Status do Servidor
```bash
curl http://localhost:8000/health
```

### `GET /tools` — Ferramentas Disponíveis
```bash
curl http://localhost:8000/tools
```

### `DELETE /sessions/{id}` — Limpar Sessão
```bash
curl -X DELETE http://localhost:8000/sessions/a1b2c3d4-...
```

### `GET /docs` — Swagger UI (automático pelo FastAPI)

## 🛠️ Ferramentas do Agente

### Leitura (Zabbix) — Sem efeito colateral
| Ferramenta | Quando usar |
|------------|-------------|
| `get_client_status` | **Sempre primeiro.** Verifica se o host está no monitoramento. |
| `get_active_incidents` | Busca alertas ativos (triggers disparadas). |
| `check_optical_signal` | Para clientes de fibra com instabilidade/lentidão. |

### Execução (Ansible) — Com impacto no serviço
| Ferramenta | Impacto | Quando usar |
|------------|---------|-------------|
| `run_router_diagnostic` | Nenhum | Diagnóstico avançado antes de ações. |
| `reset_pppoe_connection` | ~15s down | Sessão PPPoE travada, sinal OK. |
| `reboot_client_cpe` | ~5min down | **Último recurso.** Sempre avisar o cliente. |

## 🔄 Fluxo de Atendimento (ReAct Loop)

```
Cliente: "Sem internet"
   ↓
[REASON] Preciso verificar o host no Zabbix primeiro
   ↓
[ACT] get_client_status("joao-silva")
   ↓
[REASON] Host encontrado. Verificar incidentes ativos.
   ↓
[ACT] get_active_incidents("joao-silva")
   ↓
[REASON] Alerta: "PPPoE session down". Tentar reset.
   ↓
[ACT] reset_pppoe_connection("10.0.1.50")
   ↓
[REASON] Reset bem-sucedido. Confirmar com o cliente.
   ↓
"Reiniciei sua conexão. Tente agora. Funcionou?"
```

## 🔐 Segurança

- **Zabbix**: Crie um usuário read-only dedicado ao agente. Nunca use `Admin`.
- **Ansible**: Use SSH com chaves, não senhas. Configure `become` apenas onde necessário.
- **GROQ_API_KEY**: Nunca commite no Git. Use sempre o `.env`.
- **CORS**: Em produção, especifique os domínios exatos em `CORS_ORIGINS`.

## 🧪 Testando com curl (exemplos reais)

```bash
# Turno 1: Cliente reporta problema
SESSION=$(curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Oi, minha internet caiu de repente"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Turno 2: Cliente informa o ID do equipamento
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Meu equipamento é onu-cliente-bairro-01\", \"session_id\": \"$SESSION\"}"
```
