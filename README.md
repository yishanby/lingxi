# SillyTavern-Feishu Bridge

A lightweight roleplay chat service that bridges [SillyTavern](https://github.com/SillyTavern/SillyTavern) character cards with [Feishu](https://www.feishu.cn/) (Lark) bots. Manage characters and world books via a web UI, then chat in-character through Feishu groups — each character gets its own group session.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Web UI    │─────▶│  Core API    │◀─────│  Feishu Bot │
│  (Browser)  │      │  (FastAPI)   │      │  (Webhook)  │
└─────────────┘      └──────┬───────┘      └─────────────┘
                            │
                     ┌──────┴───────┐
                     │  LLM Backend │
                     │ (OpenAI/etc) │
                     └──────────────┘
```

## Features

- **Character Management** – Create, edit, delete, and import SillyTavern character cards (JSON or PNG with embedded metadata)
- **World Books** – Create and import SillyTavern world books with keyword-triggered entries
- **Multi-Backend LLM** – Configure OpenAI, Anthropic, or any OpenAI-compatible API
- **Prompt Assembly** – Full SillyTavern-style prompt building: system prompt, personality, scenario, world book entries (keyword-matched), example dialogues, chat history
- **Feishu Integration** – Webhook-based bot that handles group events, interactive character selection cards, and in-character replies
- **Web UI** – Dark-themed SPA for managing everything, with built-in chat for testing sessions
- **SQLite Storage** – Zero-config database, file-based

## MD Memory V2

`data/memory/<session-id>/chat.md` is the authoritative conversation history.
SQLite continues to store session and character metadata, but active chat turns are
read from and committed to Markdown instead of `sessions.messages`.

Long-term facts (`memory.md`), current story state (`story_state.md`), the overall
plot recap (`summary.md`), episodes (`episodes/*.md`), character profiles
(`characters/*.md`), and assets (`assets.md`) remain stored as Markdown files. The
overall plot summary does not decide whether a storyline is complete; no completion
decision is required or tracked. `rag/index.json` is a disposable derived index and
can be rebuilt from the Markdown source.

Memory stages use checkpoints in `memory_state.md`. If a background update fails or
the process restarts, the same unprocessed message range is retried without changing
the committed chat history. Web chat, HTTP Feishu events, and the Feishu WebSocket
worker all share this session API and Markdown-backed history.

The V2 runtime can be tuned with these environment variables. An interval “message”
means one user or assistant record in `chat.md`, so one complete turn normally counts
as two messages. Setting `MEMORY_V2_ENABLED=false` disables the checkpointed
background pipeline; it does not switch authoritative chat history back to SQLite.

| Variable | Default | Purpose |
|---|---:|---|
| `MEMORY_V2_ENABLED` | `true` | Enable the checkpointed Markdown memory pipeline |
| `STORY_STATE_INTERVAL_MESSAGES` | `2` | Messages between current-story-state updates |
| `MEMORY_EXTRACT_INTERVAL_MESSAGES` | `10` | Messages between long-term memory extractions |
| `EPISODE_SIZE_MESSAGES` | `20` | Messages grouped into each episode |
| `RAG_INDEX_INTERVAL_MESSAGES` | `10` | Messages between derived RAG index updates |
| `ASSETS_INTERVAL_MESSAGES` | `10` | Messages between asset-state updates |
| `MEMORY_BACKUP_COUNT` | `3` | Number of memory backups retained |
| `TOTAL_TOKEN_BUDGET` | `40000` | Hard limit for the assembled prompt context |
| `MIN_RECENT_MESSAGES` | `4` | Minimum recent messages protected in context |
| `SUMMARY_MAX_TOKENS` | `3000` | Token limit for the overall plot recap |
| `OUTPUT_RETRY_COUNT` | `1` | Retry allowance for guarded non-stream output |
| `STREAM_GUARD_CHARS` | `512` | Initial streamed text held for refusal checks |

The roleplay policy remains enabled: every model reply begins with the exact fixed
opening `我将继续这个奇幻又荒诞的故事。`; adult narrative priming remains active for
consenting-adult fiction; and the no-refusal instruction continues to require the
model to stay in the fictional roleplay. Output guards retry refusals, and only a
successfully completed user/assistant pair is committed to `chat.md`; a failed or
partial model response leaves neither half of the new turn behind.

## Quick Start

### 1. Prerequisites

- Python 3.11+

### 2. Install

```bash
# Clone and enter the project
cd sillytavern-feishu

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your settings
```

At minimum, configure an LLM backend:

| Variable | Description |
|---|---|
| `DEFAULT_LLM_PROVIDER` | `openai`, `anthropic`, or `custom` |
| `DEFAULT_LLM_API_KEY` | Your API key |
| `DEFAULT_LLM_MODEL` | Model name (e.g. `gpt-4o`, `claude-sonnet-4-6`) |
| `DEFAULT_LLM_BASE_URL` | API base URL |

For Feishu integration (optional):

| Variable | Description |
|---|---|
| `FEISHU_APP_ID` | Feishu app ID from open platform |
| `FEISHU_APP_SECRET` | Feishu app secret |
| `FEISHU_VERIFICATION_TOKEN` | Event subscription verification token |
| `FEISHU_ENCRYPT_KEY` | Event subscription encrypt key |

### 4. Run

```bash
# Development (with auto-reload)
python -m app.main

# Or directly with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** for the web UI.

## API Endpoints

### Characters
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/characters` | Create character |
| `GET` | `/api/characters` | List characters (supports `?search=`) |
| `GET` | `/api/characters/:id` | Get character |
| `PUT` | `/api/characters/:id` | Update character |
| `DELETE` | `/api/characters/:id` | Delete character |
| `POST` | `/api/characters/import` | Import ST card (JSON/PNG) |

### World Books
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/worldbooks` | Create world book |
| `GET` | `/api/worldbooks` | List world books |
| `PUT` | `/api/worldbooks/:id` | Update world book |
| `POST` | `/api/worldbooks/import` | Import ST world book |

### Sessions
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sessions` | Create session |
| `GET` | `/api/sessions` | List sessions |
| `GET` | `/api/sessions/:id` | Get session with history |
| `DELETE` | `/api/sessions/:id` | Delete session |
| `POST` | `/api/sessions/:id/message` | Send message (chat) |

### LLM Backends
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/backends` | Create backend |
| `GET` | `/api/backends` | List backends |
| `GET` | `/api/backends/:id` | Get backend |
| `PUT` | `/api/backends/:id` | Update backend |
| `DELETE` | `/api/backends/:id` | Delete backend |

### Feishu
| Method | Path | Description |
|---|---|---|
| `POST` | `/api/feishu/webhook` | Feishu event webhook |

## Feishu Bot Setup

1. Create an app on [Feishu Open Platform](https://open.feishu.cn/)
2. Enable **Bot** capability
3. Set up **Event Subscription** (HTTP mode) pointing to `https://your-host/api/feishu/webhook`
4. Subscribe to events:
   - `im.chat.member.bot.added_v1` – bot added to group
   - `im.message.receive_v1` – message received
5. Add the bot to a Feishu group
6. The bot will send a character selection card — pick a character to start

### Bot Commands
| Command | Description |
|---|---|
| `/reset` | Clear chat history |
| `/switch` | Switch to a different character |
| `/info` | Show current session info |

## Importing SillyTavern Cards

### Character Cards
- **JSON**: Standard SillyTavern character card format (V1 or V2)
- **PNG**: Character card with embedded JSON in the `chara` tEXt chunk (base64-encoded)

Upload via the web UI "Import Card" button or `POST /api/characters/import` with a multipart file.

### World Books
- **JSON**: SillyTavern world book format with `entries` object

Upload via the web UI "Import" button or `POST /api/worldbooks/import` with a multipart file.

## Project Structure

```
sillytavern-feishu/
  app/
    main.py              # FastAPI app entry
    config.py            # Settings (from .env)
    database.py          # SQLite/SQLAlchemy setup
    models/
      tables.py          # SQLAlchemy ORM models
    schemas/
      api.py             # Pydantic request/response schemas
    routers/
      characters.py      # Character CRUD + import
      worldbooks.py      # WorldBook CRUD + import
      sessions.py        # Session management + chat
      backends.py        # LLM backend config
      feishu.py          # Feishu webhook handler
    services/
      llm.py             # LLM API abstraction
      prompt.py          # Prompt assembly engine
      character_import.py  # ST card parser
      feishu_client.py   # Feishu API client
    static/
      index.html         # Web UI (Alpine.js SPA)
  requirements.txt
  .env.example
  README.md
```

## Tech Stack

- **Python 3.11+** with **FastAPI** + **uvicorn**
- **SQLite** via **SQLAlchemy** (async with aiosqlite)
- **httpx** for HTTP calls (LLM APIs, Feishu APIs)
- **Pydantic** for data validation
- **Alpine.js** for the web UI (zero build step)
