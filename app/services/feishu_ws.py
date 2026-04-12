"""Feishu WebSocket launcher — spawns the WS worker as a subprocess.

This avoids event-loop conflicts between lark-oapi SDK and uvicorn.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

_ws_process: Optional[subprocess.Popen] = None


def start_ws_client():
    """Start the Feishu WebSocket worker as a separate process."""
    global _ws_process

    if not settings.feishu_app_id or not settings.feishu_app_secret:
        logger.warning("Feishu app_id/app_secret not configured — WebSocket client not started")
        return

    worker_path = Path(__file__).parent / "feishu_ws_worker.py"
    project_root = Path(__file__).parent.parent.parent

    _ws_process = subprocess.Popen(
        [sys.executable, str(worker_path)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    logger.info(f"🔌 Feishu WebSocket worker started (PID {_ws_process.pid})")

    # Start a reader thread to forward logs
    import threading

    def _log_reader():
        if _ws_process and _ws_process.stdout:
            for line in iter(_ws_process.stdout.readline, b""):
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.info(f"[feishu-ws] {text}")

    threading.Thread(target=_log_reader, daemon=True, name="feishu-ws-log").start()


def stop_ws_client():
    """Stop the WebSocket worker process."""
    global _ws_process
    if _ws_process:
        logger.info("Stopping Feishu WebSocket worker...")
        _ws_process.terminate()
        try:
            _ws_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ws_process.kill()
        _ws_process = None


def set_session_factory(factory):
    """No-op — subprocess version doesn't need the session factory."""
    pass
