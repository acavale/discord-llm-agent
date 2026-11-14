# Deploying

The agent is a single long-running Python process plus an optional nightly batch job. Any always-on
machine works; the notes below assume a local model served by Ollama.

## 0. Discord application (one time)

In the Discord Developer Portal, under **Bot**, enable the privileged intents the agent uses:
**Message Content** (to read messages), **Server Members** (member lookups), and **Presence**
(presence-triggered posts). Invite the bot with permission to read history and send messages in the
channels you allowlist.

## 1. Model server

Install [Ollama](https://ollama.com) and pull a tool-calling instruct model. A mixture-of-experts model
is a good fit for CPU-heavy hosts: only the active experts are read per token, so generation speed is
set by active parameters, not total parameters.

```bash
ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M
```

**Serve it with an explicit context window.** The server default is 4096 tokens, and a request whose
prompt exceeds it is rejected outright. Persona + memory + transcript + tool results passes that
easily, so without this step most real requests fail while short ones succeed, which looks like the
model ignoring its context. Create a derived model:

```
# Modelfile
FROM qwen3:30b-a3b-instruct-2507-q4_K_M
PARAMETER num_ctx 16384
```

```bash
ollama create agent-model -f Modelfile
```

Set `llm.ollama.model: agent-model` in `config.yaml`.

Useful server environment variables:

| Variable | Why |
|---|---|
| `OLLAMA_KEEP_ALIVE=-1` | Keep the model resident. The OpenAI-compatible `/v1` endpoint ignores per-request `keep_alive`, so this has to be set on the server. |
| `OLLAMA_HOST=127.0.0.1:11434` | Bind locally. Ollama has no authentication; only expose it on a trusted private network. |

A cold load of a 20 GB model takes tens of seconds. `scripts/ollama_warm.ps1` (Windows) shows the
pattern for preloading it at boot so the first real request does not pay that cost.

## 2. Configuration

```bash
cp .env.example .env                       # DISCORD_TOKEN, STORAGE_ROOT
mkdir -p "$STORAGE_ROOT"
cp config.example.yaml "$STORAGE_ROOT/config.yaml"   # guild ID, owner ID, allowlisted channels
python -m discord_agent --check            # offline preflight; exits non-zero on any missing piece
```

Use a separate bot token, test server, and storage folder for development. Two processes holding the
same bot token will both answer.

## 3. Run as a service

Run `python -m discord_agent` under whatever supervisor the host provides (systemd, launchd, Windows
Task Scheduler, a container restart policy) with restart-on-failure. Two things to get right:

- **Dependencies must be visible to the service account.** Packages installed with `pip --user` are
  not on the path of a system service; install into the interpreter the service actually uses.
- **Keep the data on local disk.** A headless service may not see a user's mounted cloud drive, and a
  live SQLite index inside a synced folder invites corruption.

## 4. Nightly learning (optional)

`scripts/enrich_from_logs.py` reads the full message archive, including read-only channels, and records
durable per-member facts. Schedule it nightly, then restart the agent so it loads the new facts:

```bash
python scripts/enrich_from_logs.py --days 7 --max-msgs 80 --min-msgs 3
```

## 5. Checking on it

| Tool | Use |
|---|---|
| `python scripts/selftest.py` | Offline regression suite. |
| `python scripts/behavior_check.py` | Probe the live model with the production prompt. |
| `python scripts/ollama_diag.py` | Where latency goes: load, prefill, generation, time to first token. |
| `python scripts/log_review.py 24 <archive.jsonl>` | Recent transcript, fabricated-link audit, repetition report. |

Watch the agent's log for HTTP errors from the model endpoint. A steady trickle of 400s usually means
prompts are exceeding the served context window.
