"""Script de diagnóstico — mostra as tags reais de um host no Zabbix."""
import os, json, requests
from pathlib import Path

# Carrega .env manualmente
env_path = Path(__file__).parent / ".env"
env = {}
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()

ZABBIX_URL   = env.get("ZABBIX_URL")
ZABBIX_TOKEN = env.get("ZABBIX_TOKEN")
HOST_NAME    = "LINK14"   # ← troque se quiser testar outro host

resp = requests.post(
    ZABBIX_URL,
    json={
        "jsonrpc": "2.0", "id": 1,
        "method": "host.get",
        "params": {
            "output": ["hostid", "host", "name"],
            "selectTags": "extend",
            "search": {"host": HOST_NAME, "name": HOST_NAME},
            "searchByAny": True,
        }
    },
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ZABBIX_TOKEN}"
    },
    timeout=10
)

hosts = resp.json().get("result", [])
if not hosts:
    print(f"❌ Host '{HOST_NAME}' não encontrado no Zabbix.")
else:
    for h in hosts:
        print(f"\n✅ Host encontrado: {h['name']} (ID: {h['hostid']})")
        tags = h.get("tags", [])
        if tags:
            print("   Tags cadastradas:")
            for t in tags:
                print(f"     tag='{t['tag']}'  valor='{t['value']}'")
        else:
            print("   ⚠️  Nenhuma tag cadastrada neste host.")
