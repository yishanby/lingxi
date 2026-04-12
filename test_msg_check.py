import lark_oapi as lark
from lark_oapi.event.dispatcher_handler import EventDispatcherHandler
from lark_oapi.ws import Client as WsClient

APP_ID = "cli_a95190f8f6b8dbd2"
APP_SECRET = "qVztv5AHZnUNjo2EklYLLc5otR6Zu7oc"

def _on_message(data):
    print(">>> GOT im.message.receive_v1")
    print(data)

def _on_entered(data):
    print(">>> GOT bot_p2p_chat_entered_v1")
    print(data)

def _on_bot_added(data):
    print(">>> GOT im.chat.member.bot.added_v1")
    print(data)

def _on_card(data):
    print(">>> GOT card action")
    print(data)

print("worker build = 2026-04-12-msg-check-v2")

event_handler = (
    EventDispatcherHandler.builder("", "")
    .register_p2_im_chat_member_bot_added_v1(_on_bot_added)
    .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_on_entered)
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

ws_client.start()
