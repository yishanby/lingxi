"""Minimal Feishu WS test — just print any event received."""
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient

APP_ID = "cli_a95190f8f6b8dbd2"
APP_SECRET = "qVztv5AHZnUNjo2EklYLLc5otR6Zu7oc"


def _on_bot_added(data):
    print(f"\n{'='*60}")
    print(f"[BOT_ADDED] Received!")
    print(f"  data type: {type(data)}")
    print(f"  data: {data}")
    try:
        print(f"  chat_id: {data.event.chat_id}")
    except Exception as e:
        print(f"  error getting chat_id: {e}")
    print(f"{'='*60}\n")


def _on_message(data):
    print(f"\n{'='*60}")
    print(f"[MESSAGE] Received!")
    print(f"  data type: {type(data)}")
    print(f"  data: {data}")
    try:
        msg = data.event.message
        print(f"  chat_id: {msg.chat_id}")
        print(f"  msg_type: {msg.message_type}")
        print(f"  content: {msg.content}")
    except Exception as e:
        print(f"  error: {e}")
    print(f"{'='*60}\n")


def _on_card(data):
    print(f"\n{'='*60}")
    print(f"[CARD_ACTION] Received!")
    print(f"  data type: {type(data)}")
    print(f"  data: {data}")
    print(f"{'='*60}\n")


def _on_any(data):
    """Catch-all for custom events."""
    print(f"\n{'='*60}")
    print(f"[ANY_EVENT] Received!")
    print(f"  data: {data}")
    print(f"{'='*60}\n")


print(f"App ID: {APP_ID}")
print(f"Connecting...")

event_handler = (
    EventDispatcherHandler.builder("", "")
    .register_p2_im_chat_member_bot_added_v1(_on_bot_added)
    .register_p2_im_message_receive_v1(_on_message)
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

print("Starting WS client...")
ws_client.start()
