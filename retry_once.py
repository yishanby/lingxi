import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = "http://127.0.0.1:8000"

# Find active session in the user's chat
sessions = httpx.get(f"{BASE}/api/sessions?lite=true").json()
active = None
for s in sessions:
    if s.get("feishu_chat_id") == "oc_e451cbdeb40651591b3173a90ebdd750" and s.get("status") == "active":
        active = s
        break

if not active:
    print("No active session found")
else:
    print(f"Session #{active['id']}, sending /retry...")
    r = httpx.post(f"{BASE}/api/sessions/{active['id']}/message", json={"content": "/retry"}, timeout=60)
    data = r.json()
    messages = data.get("messages", [])
    if messages and messages[-1].get("role") == "assistant":
        print(f"Retry response (first 200): {messages[-1]['content'][:200]}")
    else:
        print(f"Response: {data}")
