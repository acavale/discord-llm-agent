# Discord LLM Agent

A Discord bot built around an LLM agent running on a local open-weight model (via Ollama). It reads the
server's conversation, decides on its own when to say something, keeps notes on the people it talks to,
and has a few tools: web search, message recall, member memory, polls, a notes knowledge base, and image
captioning.

This is a public copy of a private bot I run for a group of friends. The persona, server, channel IDs,
and member data are placeholders.

## How it works

```
Discord gateway
   |
   v
on_message --> archive (JSONL) --> local SQLite index      (every channel)
   |
   +-- read-only channel? --> learn (reflection) --> stop  (never posts)
   |
   v
speak gate (debounce, chime-in policy, mentions)            gate.py
   |
   v
prompt builder: persona + member names + context            context.py
   |
   v
agent loop: LLM <-> tool registry                           agent.py, tools/
   |
   v
output checks (tool-call leaks, repeats, echoes, bans)      llm.py, discord_app.py
   |
   v
post to an allowlisted channel

Scheduler: time- and presence-based posts                   scheduler.py
Nightly batch pass over the archive to learn member facts   scripts/enrich_from_logs.py
```

- One agent loop and one tool registry. Tools register through a small plugin contract.
- Any OpenAI-compatible endpoint works. Switching between the local model and a hosted one is one line
  of config.
- Memory is an append-only message archive, a local index for recall, per-member facts (written during
  conversation and by the nightly pass), and a notes folder.

## Guardrails

The bot can read every channel, search the web, remember people, and post without being asked, so a lot
of the code is about limiting what it does. Most of these limits are enforced in code rather than in the
prompt: with a small local model, rules written only in the prompt were not reliable.

| Guardrail | How | Where |
|---|---|---|
| Reads everywhere, posts only in allowlisted channels | A hard check after archiving. Mentions and replies elsewhere are read and learned from, never answered. | `discord_app.py` |
| Silence is the default | Replies go through a debounce and a chime-in policy, and the agent has a `stay_quiet` tool. | `gate.py`, `agent.py` |
| Chat history is data, not instructions | Archived messages, other channels, and reflection input are labeled as data in the prompt. | `context.py`, `reflection.py` |
| Owner-only commands | Admin commands check the owner's Discord ID. Owner directives go in the system prompt with rules against repeating them. | `discord_app.py` |
| Search budget | At most 2 web searches per turn, even if the model asks for several at once. Extra calls are refused, and once the budget is spent the tool is removed from the request. | `agent.py` |
| Search scope | Queries that contain a known member's name or alias are refused, so member data doesn't go to a search engine. Avoiding in-jokes is still only a prompt rule. | `tools/web.py` |
| Member memory on demand | Only names and aliases go in every prompt; full notes are fetched through a tool. | `memory.py`, `tools/memory_tools.py` |
| Output checks | Tool calls the model wrote as plain text are parsed and removed, `<think>` blocks are stripped, near-duplicate posts and verbatim echoes of a user are dropped, and banned phrases trigger one retry and then silence. | `llm.py`, `discord_app.py` |
| Greeting limit | At most one unprompted greeting per person per 24 hours, saved to disk so a restart doesn't reset it. | `scheduler.py` |
| Tools only read public channels | `recall`, `look_at_channel`, and the cross-channel context only see channels visible to `@everyone`. The bot can usually read more than a member can, so this keeps anyone from using it to read a channel they can't see. | `tools/`, `messages.py` |
| DMs only from the server | The bot answers DMs from the owner and members of its server and ignores anyone else. Tools in a DM follow the same public-channel rule. | `discord_app.py` |
| No mass pings | Messages are sent with `@everyone`, `@here`, role, and user mentions disabled, so generated text can't ping anyone. | `discord_app.py` |
| Size limits | Notes, member facts, quirks, and recall results are capped, and tool errors don't pass exception text back to the model. | `tools/` |
| Local data | Chat runs on the local model and the index stays on local disk. Image captioning uses a hosted model. | `llm.py`, `store.py` |

## Running it

Requires Python 3.10+ and either a local [Ollama](https://ollama.com) server or an OpenAI-compatible API.

```bash
pip install -r requirements.txt
cp .env.example .env                 # Discord bot token, storage folder
cp config.example.yaml config.yaml   # guild and channel IDs, model, persona (place it in the storage folder)

python -m discord_agent --check      # offline config check, no Discord connection
python -m discord_agent              # run the bot
```

If you use Ollama, serve the model with an explicit context window (see Notes). Deployment details are
in [`docs/DEPLOY.md`](docs/DEPLOY.md).

## Tests and tools

| Script | Purpose |
|---|---|
| `scripts/selftest.py` | Offline tests (no Discord, no model): speak gate, debounce, presence scheduling, greeting limit, tool-call parsing, search budget, repeat/echo/ban checks, channel scope, mention and DM limits, write caps, member-name search refusal. |
| `scripts/behavior_check.py` | Runs the real prompt against a live model and checks a few behaviors. |
| `scripts/inject_check.py` | Checks that owner directives shape a message without being repeated. |
| `scripts/smoke_provider.py` | Quick check of a model provider: tone, tool calls, text-leak detection. |
| `scripts/ollama_diag.py` | Breaks down response latency (load, prompt processing, generation). |
| `scripts/log_review.py` | Reads the archive: recent transcript, links the bot may have made up, repeated phrases. |

## Notes

- **Context window.** Ollama serves 4,096 tokens by default and rejects longer prompts with HTTP 400. For
  about eleven days most of this bot's requests failed that way while short ones went through, which
  made it look like the model was ignoring its context. The fix was serving the model with
  `num_ctx 16384`.
- **Prompt rules vs. code.** Several behaviors (not searching for members, not repeating a catchphrase,
  answering direct questions) didn't hold as prompt instructions and were moved into code checks.
- **Model.** It runs on a 30B mixture-of-experts model at Q4 on a CPU-heavy machine. Generation is
  limited by memory bandwidth, and prompt length is the main factor in response time.

## Limitations

- Prompt injection is limited, not prevented. The model still reads untrusted text; the guardrails limit
  where it can post, what tools it can reach, and what gets through the output checks.
- Member facts and notes the agent writes are fed back into later prompts, so someone can try to plant
  false facts about a person. Writes are length-capped, the agent can't create new members, and notes are
  labeled unverified when shown, but facts don't record who said them, so this is reduced, not prevented.
- The repeat and echo checks are heuristics and can miss or over-block.
- Tone and topic choice still depend on the model.
- Built for a single server.
