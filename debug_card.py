import httpx, json, sys
sys.stdout.reconfigure(encoding='utf-8')

# Simulate creating a streaming card to debug
FEISHU_BASE = "https://open.feishu.cn/open-apis"
CARDKIT_BASE = f"{FEISHU_BASE}/cardkit/v1/cards"

# Get token
from app.services.feishu_ws_worker import get_tenant_access_token, _feishu_headers

token = get_tenant_access_token()
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

card_json = {
    "schema": "2.0",
    "config": {
        "streaming_mode": True,
        "summary": {"content": "[生成中...]"},
        "streaming_config": {
            "print_frequency_ms": {"default": 50},
            "print_step": {"default": 1},
        },
    },
    "header": {
        "title": {"tag": "plain_text", "content": "测试角色"},
        "template": "violet",
    },
    "body": {
        "elements": [
            {"tag": "markdown", "content": "┃", "element_id": "content"}
        ]
    },
}

print("Step 1: Creating card...")
resp = httpx.post(
    CARDKIT_BASE,
    headers=headers,
    json={"type": "card_json", "data": json.dumps(card_json, ensure_ascii=False)},
    timeout=10,
)
data = resp.json()
print(f"  Response: code={data.get('code')}, data={json.dumps(data.get('data', {}), ensure_ascii=False)}")

if data.get("code") == 0 and data.get("data", {}).get("card_id"):
    card_id = data["data"]["card_id"]
    print(f"  card_id: {card_id}")
    
    # Try different content formats
    chat_id = "oc_e451cbdeb40651591b3173a90ebdd750"
    
    # Format 1: what we currently use
    content1 = json.dumps({"type": "card", "data": {"card_id": card_id}})
    print(f"\nStep 2a: Sending with format 1: {content1}")
    msg_resp = httpx.post(
        f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
        headers=_feishu_headers(),
        json={
            "receive_id": chat_id,
            "msg_type": "interactive",
            "content": content1,
        },
        timeout=10,
    )
    msg_data = msg_resp.json()
    print(f"  Response: code={msg_data.get('code')}, msg={msg_data.get('msg', '')}")
    
    if msg_data.get("code") != 0:
        # Format 2: try with card_id directly in content
        content2 = json.dumps({"card_id": card_id})
        print(f"\nStep 2b: Sending with format 2: {content2}")
        msg_resp2 = httpx.post(
            f"{FEISHU_BASE}/im/v1/messages?receive_id_type=chat_id",
            headers=_feishu_headers(),
            json={
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": content2,
            },
            timeout=10,
        )
        msg_data2 = msg_resp2.json()
        print(f"  Response: code={msg_data2.get('code')}, msg={msg_data2.get('msg', '')}")
