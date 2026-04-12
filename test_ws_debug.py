"""Feishu WS debug test v4 — all events registered, version stamp."""
import json
import sys
import os
import time
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("WS worker code version = 2026-04-12-v4-all-events")
print("=" * 60, flush=True)

import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient
from lark_oapi.ws.client import Frame

APP_ID = "cli_a95190f8f6b8dbd2"
APP_SECRET = "qVztv5AHZnUNjo2EklYLLc5otR6Zu7oc"


def _on_bot_added(data):
    print(f"\n>>> [BOT_ADDED] received!", flush=True)
    try:
        print(f"    chat_id: {data.event.chat_id}", flush=True)
    except Exception as e:
        print(f"    error: {e}", flush=True)

def _on_message(data):
    print(f"\n>>> [MESSAGE] im.message.receive_v1 received!", flush=True)
    print(f"    data: {data}", flush=True)
    try:
        msg = data.event.message
        print(f"    chat_id: {msg.chat_id}", flush=True)
        print(f"    content: {msg.content}", flush=True)
        print(f"    msg_type: {msg.message_type}", flush=True)
    except Exception as e:
        print(f"    error: {e}", flush=True)

def _on_entered(data):
    print(f"\n>>> [P2P_ENTERED] bot_p2p_chat_entered received!", flush=True)
    try:
        print(f"    chat_id: {data.event.chat_id}", flush=True)
    except Exception as e:
        print(f"    error: {e}", flush=True)

def _on_card(data):
    print(f"\n>>> [CARD] card_action_trigger received!", flush=True)
    print(f"    data: {data}", flush=True)


event_handler = (
    EventDispatcherHandler.builder("", "")
    .register_p2_im_chat_member_bot_added_v1(_on_bot_added)
    .register_p2_im_message_receive_v1(_on_message)
    .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_on_entered)
    .register_p2_card_action_trigger(_on_card)
    .build()
)

ws_client = WsClient(
    app_id=APP_ID,
    app_secret=APP_SECRET,
    event_handler=event_handler,
    log_level=lark.LogLevel.DEBUG,
    auto_reconnect=True,
)

# Monkey-patch to see ALL raw frames
_orig_handle = ws_client._handle_message

async def _debug_handle(msg: bytes):
    try:
        frame = Frame()
        frame.ParseFromString(msg)
        headers = {h.key: h.value for h in frame.headers}
        msg_type = headers.get('type', '?')
        msg_id = headers.get('message_id', '?')
        print(f"\n>>> FRAME method={frame.method} type={msg_type} msg_id={msg_id}", flush=True)
        if frame.payload:
            try:
                pl = frame.payload.decode('utf-8')
                # Extract event_type if present
                try:
                    d = json.loads(pl)
                    et = d.get('header', {}).get('event_type', 'N/A')
                    print(f"    event_type={et}", flush=True)
                except:
                    pass
                print(f"    payload={pl[:500]}", flush=True)
            except:
                print(f"    payload=<binary {len(frame.payload)} bytes>", flush=True)
    except Exception as e:
        print(f">>> FRAME parse error: {e}", flush=True)
    
    return await _orig_handle(msg)

ws_client._handle_message = _debug_handle

print("Starting WS client...", flush=True)
ws_client.start()
