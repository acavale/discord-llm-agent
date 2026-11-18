#!/usr/bin/env python3
"""Offline self-tests for the chime/debounce/presence logic (no network; _respond is stubbed).

Hermetic: builds a throwaway storage root from config.example.yaml plus a fixture member file, so it never
reads (or writes) a real deployment's data and passes on a fresh clone.

    python scripts/selftest.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import discord

from discord_agent.config import load_config
from discord_agent.store import Store
from discord_agent.discord_app import DiscordAgent
from discord_agent.scheduler import Scheduler
from discord_agent import agent as agentmod


def _hermetic_env() -> None:
    """Point the loader at a fresh temp storage root seeded from the example config and a fixture member
    file. Overrides any STORAGE_ROOT in the environment so tests can never touch real data."""
    root = Path(tempfile.mkdtemp(prefix="agent-selftest-"))
    shutil.copy(_REPO / "config.example.yaml", root / "config.yaml")
    (root / "server_context.json").write_text(json.dumps({"server_name": "test", "users": [
        {"name": "Jordan", "aliases": ["jordan", "casey"], "role": "Member", "traits": [], "activities": []},
        {"name": "Taylor", "aliases": ["taylor"], "role": "Member", "traits": [], "activities": []},
    ]}), encoding="utf-8")
    os.environ["STORAGE_ROOT"] = str(root)
    os.environ["LOCAL_DATA_DIR"] = str(root / "local")
    os.environ.setdefault("DISCORD_TOKEN", "selftest-dummy")
    os.environ.setdefault("XAI_API_KEY", "selftest-dummy")


async def main() -> int:
    _hermetic_env()
    cfg = load_config()
    st = Store(cfg)
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    intents.presences = True
    c = DiscordAgent(cfg, st, intents=intents)

    # name-drop detection
    assert c._is_name_drop("man agent is dumb")
    assert c._is_name_drop("shut up bot")
    assert not c._is_name_drop("just playing fortnite")
    print("name_drop_detect OK")

    # debounce coalescing: a burst becomes ONE reply; directed (allow_silence=False) wins
    calls = []

    async def fake_respond(channel, cid, allow_silence=False, consume_budget=False):
        calls.append((cid, allow_silence, consume_budget))

    c._respond = fake_respond
    c.debounce_secs = 0.3

    class Ch:
        id = 999

    ch = Ch()
    c._schedule_response(ch, "999", allow_silence=True, consume_budget=True)
    await asyncio.sleep(0.1)
    c._schedule_response(ch, "999", allow_silence=True, consume_budget=True)
    await asyncio.sleep(0.1)
    c._schedule_response(ch, "999", allow_silence=False, consume_budget=False)
    await asyncio.sleep(0.6)
    assert len(calls) == 1, f"expected 1 coalesced reply, got {len(calls)}: {calls}"
    assert calls[0][1] is False, "directed trigger should force allow_silence=False"
    print("debounce_coalesce OK", calls)

    # presence: identity groups built from server_context, with alias bridge
    groups = c._identity_groups()
    assert any("jordan" in g and "casey" in g for g in groups), "alias group missing"
    print("identity_groups OK")

    # scheduler matches case-insensitively (schedule 'Jordan' vs online token 'jordan')
    tmp = Path(tempfile.mkdtemp()) / "sched.json"
    tmp.write_text(json.dumps({"schedules": [{
        "id": "t", "type": "condition_based", "enabled": True, "channel_id": "1",
        "conditions": {"users_online": ["Jordan"]}, "cooldown_hours": 1,
    }]}))
    def rec(key, names, acts=(), platforms=("desktop",), display=None):
        return {"key": key, "display": display or key, "names": {n.lower() for n in names},
                "status": "online", "platforms": list(platforms),
                "activities": [{"type": t, "name": n} for (t, n) in acts]}

    fired = []

    async def on_intent(s, match=None):
        fired.append((s["id"], (match or {}).get("activity_name", "")))

    sc = Scheduler(tmp, on_intent=on_intent, is_channel_busy=lambda _c: False)
    sc._primed = True  # simulate a live (post-startup) state so the first tick is treated as a transition
    await sc.on_presence([rec("100", ["jordan"])])  # legacy conditions schedule, came-online trigger
    assert [f[0] for f in fired] == ["t"], f"legacy presence match failed: {fired}"
    print("presence_lowercase_match OK")

    # rich-presence: fire only when a matching activity STARTS, with type + name filters; not on stop/refire
    tmp2 = Path(tempfile.mkdtemp()) / "sched2.json"
    tmp2.write_text(json.dumps({"schedules": [{
        "id": "ftn", "type": "condition_based", "enabled": True, "channel_id": "1", "cooldown_hours": 1,
        "presence": {"users_online": [], "trigger_on_new_activity": True, "filters": {
            "activity_type": {"enabled": True, "value": "playing"},
            "activity_name_contains": {"enabled": True, "value": "fortnite"},
            "platform": {"enabled": False, "value": None}}},
    }]}))
    fired2 = []

    async def on_intent2(s, match=None):
        fired2.append((s["id"], (match or {}).get("activity_name", "")))

    sc2 = Scheduler(tmp2, on_intent=on_intent2, is_channel_busy=lambda _c: False)
    sc2._primed = True
    await sc2.on_presence([rec("200", ["taylor"], acts=[("listening", "A$AP - x")])])      # online, no fortnite
    assert fired2 == [], f"should not fire without fortnite: {fired2}"
    await sc2.on_presence([rec("200", ["taylor"], acts=[("listening", "A$AP - x"), ("playing", "Fortnite")])])
    assert [f[0] for f in fired2] == ["ftn"] and fired2[0][1] == "Fortnite", f"fortnite start: {fired2}"
    await sc2.on_presence([rec("200", ["taylor"], acts=[("playing", "Fortnite")])])        # no new act + cooldown
    assert [f[0] for f in fired2] == ["ftn"], f"should not refire: {fired2}"
    print("presence_activity_trigger OK")

    # broad greeting (trigger_on_new_activity + no filters) fires on ANY change kind; a filtered schedule
    # stays activity-specific. Checked via _match_user directly to sidestep the 1h cooldown floor.
    pcfg_broad = {"users": {"taylor"}, "new_activity": True, "activity_type": None,
                  "name_contains": None, "platform": None}
    cur_b = {"200": rec("200", ["taylor"], acts=[("playing", "Fortnite")])}
    assert sc2._match_user(pcfg_broad, cur_b, {"200": {"came_online": False,
        "new_acts": {("playing", "Fortnite")}, "new_platforms": set()}}), "broad: activity start should fire"
    assert sc2._match_user(pcfg_broad, cur_b, {"200": {"came_online": True,
        "new_acts": set(), "new_platforms": set()}}), "broad: came-online should fire"
    assert sc2._match_user(pcfg_broad, cur_b, {"200": {"came_online": False,
        "new_acts": set(), "new_platforms": set()}}) is None, "broad: no change should not fire"
    pcfg_ftn = {"users": set(), "new_activity": True, "activity_type": "playing",
                "name_contains": "fortnite", "platform": None}
    assert sc2._match_user(pcfg_ftn, cur_b, {"200": {"came_online": True,
        "new_acts": set(), "new_platforms": set()}}) is None, "filtered: came-online alone must not fire"
    print("presence_broad_and_filtered OK")

    roster = DiscordAgent._presence_roster_text([
        rec("1", ["casey"], acts=[("playing", "Fortnite")], display="casey"),
        rec("2", ["taylor"], acts=[("playing", "Fortnite")], display="taylor"),
        rec("3", ["karl"], acts=[("listening", "A$AP Mob - x")], display="karl"),
        rec("4", ["morgan"], acts=[], display="morgan"),
    ])
    assert "Fortnite: casey, taylor" in roster, roster
    assert "listening to A$AP Mob - x: karl" in roster, roster
    assert "nothing showing: morgan" in roster, roster
    print("presence_roster_text OK")

    # ONE post per user per window, whatever the trigger. A plain come-on greets at most once per the (long)
    # per-user cooldown, and a real activity start/switch NO LONGER bypasses that window (it used to fire
    # unconditionally, so anyone launching/switching games got greeted repeatedly). A stop fires nothing;
    # music starts and song-to-song churn don't fire. After the window expires, the user is eligible again.
    tmp3 = Path(tempfile.mkdtemp()) / "sched3.json"
    tmp3.write_text(json.dumps({"schedules": [{
        "id": "mon", "type": "condition_based", "enabled": True, "channel_id": "1",
        "presence": {"users_online": [], "trigger_on_new_activity": True, "per_user_cooldown": True,
                     "per_user_cooldown_minutes": 1440, "filters": {
                         "activity_type": {"enabled": False, "value": None},
                         "activity_name_contains": {"enabled": False, "value": None},
                         "platform": {"enabled": False, "value": None}}},
    }]}))
    fired3 = []

    async def on_intent3(s, match=None):
        fired3.append(((match or {}).get("user", ""), (match or {}).get("activity_name", "")))

    sc3 = Scheduler(tmp3, on_intent=on_intent3, is_channel_busy=lambda _c: False)
    sc3._primed = True
    await sc3.on_presence([rec("10", ["casey"], display="casey")])               # online idle -> greeted
    assert fired3 == [("casey", "")], fired3
    await sc3.on_presence([])                                                         # dips offline
    await sc3.on_presence([rec("10", ["casey"], display="casey")])               # back idle, in window
    assert fired3 == [("casey", "")], f"repeat come-on within window should not re-greet: {fired3}"
    await sc3.on_presence([rec("10", ["casey"], acts=[("playing", "Fortnite")], display="casey")])
    assert len(fired3) == 1, f"game start must NOT re-post inside the per-user window: {fired3}"
    await sc3.on_presence([rec("10", ["casey"], acts=[("playing", "Marvel Rivals")], display="casey")])
    assert len(fired3) == 1, f"game switch must NOT re-post inside the window: {fired3}"
    await sc3.on_presence([rec("10", ["casey"], acts=[("playing", "Marvel Rivals"),
                           ("listening", "A$AP - x")], display="casey")])           # music start (not real)
    assert len(fired3) == 1, f"music start should not fire: {fired3}"
    await sc3.on_presence([rec("10", ["casey"], acts=[("playing", "Marvel Rivals"),
                           ("listening", "A$AP - y")], display="casey")])           # song-to-song churn
    assert len(fired3) == 1, f"song churn should not fire: {fired3}"
    await sc3.on_presence([rec("10", ["casey"], display="casey")])               # game -> idle (a stop)
    assert len(fired3) == 1, f"a stop should not fire: {fired3}"
    # window expires -> eligible again (backdate the flag past the 1440-minute window)
    sc3._user_fired[("mon", "10")] = datetime.datetime.now() - datetime.timedelta(hours=25)
    await sc3.on_presence([rec("10", ["casey"], acts=[("playing", "Fortnite")], display="casey")])
    assert len(fired3) == 2 and fired3[-1] == ("casey", "Fortnite"), f"should fire after window: {fired3}"
    print("presence_one_post_per_window OK")

    # greeting probability: a plain come-on is greeted only on a winning roll (an activity start is not
    # probability-gated). A losing roll stamps nothing, so the user re-rolls on the next pop-on. Once ANY
    # post lands for a user -- greeting or activity -- the per-user window blocks further posts regardless of
    # rolls, until it expires. RNG is patched for determinism.
    import discord_agent.scheduler as schedmod
    tmpP = Path(tempfile.mkdtemp()) / "schedP.json"
    tmpP.write_text(json.dumps({"schedules": [{
        "id": "mon", "type": "condition_based", "enabled": True, "channel_id": "1",
        "presence": {"users_online": [], "trigger_on_new_activity": True, "per_user_cooldown": True,
                     "per_user_cooldown_minutes": 1440, "filters": {
                         "activity_type": {"enabled": False, "value": None},
                         "activity_name_contains": {"enabled": False, "value": None},
                         "platform": {"enabled": False, "value": None}}},
    }]}))
    firedP = []

    async def on_intentP(s, match=None):
        firedP.append(((match or {}).get("user", ""), (match or {}).get("activity_name", "")))

    scP = Scheduler(tmpP, on_intent=on_intentP, is_channel_busy=lambda _c: False, greet_probability=0.75)
    scP._primed = True
    _orig_rand = schedmod.random.random
    try:
        schedmod.random.random = lambda: 0.9                                   # losing roll (0.9 >= 0.75)
        await scP.on_presence([rec("70", ["z"], display="z")])
        await scP.on_presence([])
        await scP.on_presence([rec("70", ["z"], display="z")])
        assert firedP == [], f"losing rolls should not greet (and not stamp): {firedP}"
        await scP.on_presence([rec("70", ["z"], acts=[("playing", "Fortnite")], display="z")])
        assert firedP == [("z", "Fortnite")], f"activity fires even on a losing roll (never stamped): {firedP}"
        await scP.on_presence([])                                              # z offline (drops the game)
        schedmod.random.random = lambda: 0.1                                   # winning roll (0.1 < 0.75)
        await scP.on_presence([rec("70", ["z"], display="z")])
        assert len(firedP) == 1, f"the activity post stamped z, so no greeting inside the window: {firedP}"
        # expire the window -> a winning roll greets again
        scP._user_fired[("mon", "70")] = datetime.datetime.now() - datetime.timedelta(hours=25)
        await scP.on_presence([])
        await scP.on_presence([rec("70", ["z"], display="z")])
        assert len(firedP) == 2 and firedP[1] == ("z", ""), f"winning roll after window should greet: {firedP}"
    finally:
        schedmod.random.random = _orig_rand
    print("presence_greet_probability OK")

    # startup priming: the FIRST on_presence tick after boot records the baseline (everyone already online
    # or mid-activity) WITHOUT firing, so a restart no longer spams the channel. A later real transition fires.
    tmp4 = Path(tempfile.mkdtemp()) / "sched4.json"
    tmp4.write_text(json.dumps({"schedules": [{
        "id": "mon", "type": "condition_based", "enabled": True, "channel_id": "1",
        "presence": {"users_online": [], "trigger_on_new_activity": True, "per_user_cooldown": True,
                     "per_user_cooldown_minutes": 60, "filters": {
                         "activity_type": {"enabled": False, "value": None},
                         "activity_name_contains": {"enabled": False, "value": None},
                         "platform": {"enabled": False, "value": None}}},
    }]}))
    fired4 = []

    async def on_intent4(s, match=None):
        fired4.append((match or {}).get("user", ""))

    sc4 = Scheduler(tmp4, on_intent=on_intent4, is_channel_busy=lambda _c: False)  # _primed defaults False
    await sc4.on_presence([rec("10", ["casey"], display="casey")])             # boot: online -> prime only
    assert fired4 == [], f"startup tick must prime silently, not fire: {fired4}"
    await sc4.on_presence([rec("10", ["casey"], acts=[("playing", "Fortnite")], display="casey")])
    assert fired4 == ["casey"], f"post-startup transition should fire: {fired4}"
    print("presence_startup_priming OK")

    # noise filtering: custom-status edits and bare platform switches must NOT wake the monitor; only coming
    # online or a real activity start does.
    tmp6 = Path(tempfile.mkdtemp()) / "sched6.json"
    tmp6.write_text(json.dumps({"schedules": [{
        "id": "mon", "type": "condition_based", "enabled": True, "channel_id": "1",
        "presence": {"users_online": [], "trigger_on_new_activity": True, "per_user_cooldown": True,
                     "per_user_cooldown_minutes": 1, "filters": {
                         "activity_type": {"enabled": False, "value": None},
                         "activity_name_contains": {"enabled": False, "value": None},
                         "platform": {"enabled": False, "value": None}}},
    }]}))
    fired6 = []

    async def on_intent6(s, match=None):
        fired6.append((match or {}).get("user", ""))

    sc6 = Scheduler(tmp6, on_intent=on_intent6, is_channel_busy=lambda _c: False)  # _primed defaults False
    await sc6.on_presence([rec("10", ["b"], acts=[("custom", "vibing")], platforms=("desktop",), display="b")])
    assert fired6 == [], f"startup primes: {fired6}"
    await sc6.on_presence([rec("10", ["b"], acts=[("custom", "grinding")], platforms=("desktop",), display="b")])
    assert fired6 == [], f"custom-status edit must not fire: {fired6}"
    await sc6.on_presence([rec("10", ["b"], acts=[("custom", "grinding")], platforms=("mobile",), display="b")])
    assert fired6 == [], f"platform flip must not fire: {fired6}"
    await sc6.on_presence([rec("10", ["b"], acts=[("custom", "grinding"), ("playing", "Fortnite")],
                           platforms=("mobile",), display="b")])
    assert fired6 == ["b"], f"real activity start should fire: {fired6}"
    print("presence_noise_filtered OK")

    # presence event line: anchors a fired post on the SPECIFIC change, not a generic room summary
    ev_game = DiscordAgent._presence_event_line({"user": "taylor", "activity_name": "Fortnite", "activity_type": "playing"})
    assert "taylor" in ev_game and "Fortnite" in ev_game and "playing" in ev_game and "specifically" in ev_game, ev_game
    ev_online = DiscordAgent._presence_event_line({"user": "karl", "activity_name": "", "activity_type": ""})
    assert "karl" in ev_online and "came online" in ev_online and "room" in ev_online, ev_online
    ev_status = DiscordAgent._presence_event_line({"user": "morgan", "activity_name": "grinding", "activity_type": "custom"})
    assert "morgan" in ev_status and "came online" in ev_status and "grinding" in ev_status, ev_status
    print("presence_event_line OK")

    # send-chain step 1: the follow-up decision. Continues ONLY on an explicit keep_going; defaults to stop
    # on a plain answer, and short-circuits to stop at the cap WITHOUT asking the model.
    class _FakeLLM:
        def __init__(self, resp):
            self._resp = resp

        def complete(self, *a, **k):
            return self._resp

    keep = _FakeLLM({"content": "", "tool_calls": [{"id": "1", "name": "keep_going", "arguments": {}}]})
    stop = _FakeLLM({"content": "DONE", "tool_calls": []})
    assert await agentmod.decide_followup(keep, "sys", [{"role": "user", "content": "hi"}], 1, 3) is True
    assert await agentmod.decide_followup(stop, "sys", [{"role": "user", "content": "hi"}], 1, 3) is False
    assert await agentmod.decide_followup(keep, "sys", [], 3, 3) is False  # at cap: stop without asking
    print("followup_decision OK")

    # send-chain step 2: the _emit_chain loop. One message per turn; context rebuilt from the live transcript
    # between turns; continues on decide_followup up to the hard cap. Agent + decision are stubbed.
    class _NullCtx:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    class _FakeChannel:
        def __init__(self, cid):
            self.sent = []
            self.id = cid
            self.guild = None  # skip the archive path (no disk writes in the test)

        def typing(self):
            return _NullCtx()

        async def send(self, text):
            self.sent.append(text)

    _orig_run, _orig_decide = agentmod.run, agentmod.decide_followup
    _saved_style = c.style_samples
    c.style_samples = 0  # skip the recent_chatter DB hit during follow-up context rebuilds
    try:
        # (a) continue once then stop: two messages, in order; returns the last one
        seq_a = ["hey alex hows it going", "you been on fortnite lately?"]
        dec_a = [True, False]
        st_a = {"r": 0, "d": 0}

        async def _run_a(*a, **k):
            i = st_a["r"]; st_a["r"] += 1
            return agentmod.AgentResult(text=seq_a[i] if i < len(seq_a) else None)

        async def _dec_a(*a, **k):
            i = st_a["d"]; st_a["d"] += 1
            return dec_a[i] if i < len(dec_a) else False

        agentmod.run, agentmod.decide_followup = _run_a, _dec_a
        ch_a = _FakeChannel(4242)
        last = await c._emit_chain(ch_a, str(ch_a.id), "sys", [{"role": "user", "content": "yo"}],
                                   allow_silence=False, consume_budget=False, max_chain=3)
        assert ch_a.sent == seq_a, ch_a.sent
        assert last == seq_a[-1], last

        # (b) hard cap stops the chain even when the model always wants to continue
        seq_b = ["m1", "m2", "m3", "m4"]
        st_b = {"r": 0}

        async def _run_b(*a, **k):
            i = st_b["r"]; st_b["r"] += 1
            return agentmod.AgentResult(text=seq_b[i] if i < len(seq_b) else None)

        async def _always(*a, **k):
            return True

        agentmod.run, agentmod.decide_followup = _run_b, _always
        ch_b = _FakeChannel(4243)
        await c._emit_chain(ch_b, str(ch_b.id), "sys", [{"role": "user", "content": "yo"}], max_chain=2)
        assert ch_b.sent == ["m1", "m2"], ch_b.sent

        # (c) a human speaking mid-chain (cancel event set) ends the chain after the current message
        st_c = {"r": 0}
        ch_c = _FakeChannel(4244)

        async def _run_c(*a, **k):
            st_c["r"] += 1
            ev = c._chain_cancel.get(str(ch_c.id))  # simulate on_message firing right after the first send
            if ev is not None:
                ev.set()
            return agentmod.AgentResult(text=f"msg{st_c['r']}")

        agentmod.run, agentmod.decide_followup = _run_c, _always
        await c._emit_chain(ch_c, str(ch_c.id), "sys", [{"role": "user", "content": "yo"}], max_chain=3)
        assert ch_c.sent == ["msg1"], ch_c.sent
    finally:
        agentmod.run, agentmod.decide_followup = _orig_run, _orig_decide
        c.style_samples = _saved_style
    print("emit_chain OK")

    # message-count weighting: weights load (3, positive); the cap merges any excess into the last message
    assert len(c.msg_count_weights) == 3 and sum(c.msg_count_weights) > 0, c.msg_count_weights
    assert c._split_into_messages("a\n---\nb\n---\nc", cap=1) == ["a\n\nb\n\nc"]
    assert c._split_into_messages("a\n---\nb\n---\nc", cap=2) == ["a", "b\n\nc"]
    assert c._split_into_messages("a\n---\nb\n---\nc", cap=3) == ["a", "b", "c"]
    assert "SINGLE" in c._msg_count_directive(1) and "EXACTLY 3" in c._msg_count_directive(3)
    print("message_count_weighting OK")

    # member memory dedupe: an exact restatement or a redundant substring is NOT re-appended; a genuinely
    # new fact is; an unknown name creates a member. Guards against bloat once reflection writes regularly.
    from discord_agent.memory import MemberMemory
    mm_path = Path(tempfile.mkdtemp()) / "server_context.json"
    mm_path.write_text(json.dumps({"server_name": "t", "users": [
        {"name": "Taylor", "aliases": ["taylor"], "role": "Member", "traits": ["plays Terraria"], "activities": []}
    ]}))
    mm = MemberMemory(mm_path)
    assert mm.note("taylor", "plays Terraria").startswith("("), "exact dup should be skipped"
    assert mm.note("Taylor", "Terraria").startswith("("), "substring of an existing trait should be skipped"
    assert mm.note("taylor", "into chess").startswith("Noted"), "a new fact should be recorded"
    assert mm.data["users"][0]["traits"] == ["plays Terraria", "into chess"], mm.data["users"][0]["traits"]
    assert mm.note("Karl", "shares music").startswith("Added"), "a new member should be created"
    print("member_note_dedupe OK")

    # reflection: strict-JSON extraction from a stubbed LLM, then apply through the deduping note. Fenced
    # output is tolerated; junk/empty/malformed output yields no facts and never raises.
    from discord_agent import reflection as reflectmod

    class _JsonLLM:
        def __init__(self, content):
            self._c = content

        def complete(self, *a, **k):
            return {"content": self._c, "tool_calls": []}

    good = _JsonLLM('{"facts": [{"name": "taylor", "fact": "codes in C"}, '
                    '{"name": "taylor", "fact": "into chess"}]}')  # 2nd already known -> deduped on apply
    facts = reflectmod.extract_member_facts(good, "taylor: been writing C all day", "known: into chess", ["taylor"])
    assert [f["fact"] for f in facts] == ["codes in C", "into chess"], facts
    applied = reflectmod.apply_facts(mm, facts)
    assert applied == ["Noted about Taylor: codes in C"], applied  # the known one was skipped by note()
    fenced = _JsonLLM('```json\n{"facts": [{"name": "karl", "fact": "makes playlists"}]}\n```')
    assert reflectmod.extract_member_facts(fenced, "karl: heres a playlist", "", ["karl"])[0]["fact"] == "makes playlists"
    assert reflectmod.extract_member_facts(_JsonLLM("i cannot help"), "x: hi", "", ["x"]) == []
    assert reflectmod.extract_member_facts(_JsonLLM(""), "x: hi", "", ["x"]) == []
    assert reflectmod.extract_member_facts(_JsonLLM('{"facts": "nope"}'), "x: hi", "", ["x"]) == []
    assert reflectmod.extract_member_facts(_JsonLLM('{"facts": []}'), "", "", ["x"]) == []  # empty evidence
    print("reflection_extract_apply OK")

    # owner !reflect command wiring: seed a human line, stub the reflection pass, assert it reports back.
    from discord_agent.context import Entry as _Entry
    _rc_replies = []

    class _NullCtx2:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    class _ReflectChan:
        def __init__(self, cid):
            self.id = cid
            self.name = "chat"

        def typing(self):
            return _NullCtx2()

    class _ReflectMsg:
        def __init__(self, cid):
            self.channel = _ReflectChan(cid)

        async def reply(self, text):
            _rc_replies.append(text)

    c.transcripts.add("88", _Entry(user="karl", content="new soundcloud drop tonight"))
    _orig_rac = c._reflect_after_chain

    async def _fake_rac(cid):
        return ["Noted about Karl: shares SoundCloud drops"]

    c._reflect_after_chain = _fake_rac
    try:
        await c._cmd_reflect(_ReflectMsg("88"))
        assert _rc_replies and _rc_replies[0].startswith("recorded:") and "SoundCloud" in _rc_replies[0], _rc_replies
    finally:
        c._reflect_after_chain = _orig_rac
    print("reflect_command OK")

    # quirks surface the most RECENT, not the oldest (uses is never incremented, so ranking by it would
    # bury newly-formed quirks). Add >12 and assert the newest show and the oldest are dropped.
    from discord_agent.memory import Quirks
    q = Quirks(Path(tempfile.mkdtemp()) / "quirks.json", max_quirks=24)
    for i in range(14):
        q.data["quirks"].append({"text": f"quirk{i}", "created": 1000 + i, "uses": 0})
    shown = {ln[2:] for ln in q.format_for_prompt().splitlines() if ln.startswith("- ")}
    assert "quirk13" in shown and "quirk2" in shown, shown          # newest 12 shown
    assert "quirk0" not in shown and "quirk1" not in shown, shown   # oldest two buried
    print("quirks_recency OK")

    # provider switch: LLM builds the right endpoint URL per provider and merges provider-specific payload
    # extras (Ollama keep_alive) without polluting the Grok path.
    from discord_agent.llm import LLM as _LLM
    _g = _LLM(api_key="x", model="grok-4.20-0309-reasoning")
    assert _g.url == "https://api.x.ai/v1/chat/completions", _g.url
    assert _g.extra_payload == {}, _g.extra_payload
    _o = _LLM(api_key="ollama", model="qwen3:30b-a3b-instruct-2507-q8_0",
              base_url="http://localhost:11434/v1", extra_payload={"keep_alive": -1})
    assert _o.url == "http://localhost:11434/v1/chat/completions", _o.url
    assert _o.extra_payload == {"keep_alive": -1}, _o.extra_payload
    print("llm_provider_switch OK")

    # tool-call recovery: a local model that emits a Hermes <tool_call> block (or a bare JSON tool object)
    # as TEXT gets it recovered into a structured call AND stripped from the content, so the tool runs and
    # the raw call never leaks into a channel. This is the fix for the Qwen3-via-Ollama leak.
    from discord_agent.llm import _recover_tool_calls_from_text as _rtc
    _tc, _clean = _rtc('sup gamers\n<tool_call>\n{"name": "web_search", "arguments": {"query": "fortnite update"}}\n</tool_call>')
    assert len(_tc) == 1 and _tc[0]["name"] == "web_search" and _tc[0]["arguments"]["query"] == "fortnite update", _tc
    assert "<tool_call>" not in _clean and "web_search" not in _clean, repr(_clean)
    _tc2, _clean2 = _rtc('{"name": "recall", "arguments": {"query": "lebron"}}')
    assert _tc2 and _tc2[0]["name"] == "recall" and _clean2 == "", (_tc2, _clean2)
    _tc3, _clean3 = _rtc("just a normal banter message with no tools")
    assert _tc3 == [] and _clean3 == "just a normal banter message with no tools", (_tc3, _clean3)
    # positional narration of a known tool (the DM leak form) is stripped whole-line, keeping real text
    _tc4, _clean4 = _rtc('huh weird\nremember_member("alex", "likes glitch spirits")', frozenset({"remember_member"}))
    assert _tc4 == [] and "remember_member(" not in _clean4 and _clean4.strip() == "huh weird", (_tc4, _clean4)
    print("toolcall_recovery OK")

    # search-spiral backstop: web_search is dropped from the toolset after _SEARCH_CAP calls so the loop
    # can't chain endless failing searches. Fake LLM searches whenever it's allowed; assert it stops at cap.
    from discord_agent import agent as _agentmod
    from discord_agent.agent import _SEARCH_CAP

    class _SearchLLM:
        def complete(self, messages, tools=None, temperature=None, tool_choice="auto"):
            names = {t.get("function", {}).get("name") for t in (tools or [])}
            if "web_search" in names:
                return {"content": "", "tool_calls": [{"id": "", "name": "web_search", "arguments": {"query": "x"}}]}
            return {"content": "done searching", "tool_calls": []}

    _dispatch_calls = []
    _orig_dispatch = _agentmod.toolreg.dispatch

    async def _fake_dispatch(name, args, ctx):
        _dispatch_calls.append(name)
        return "no results"

    _agentmod.toolreg.dispatch = _fake_dispatch
    try:
        _res = await _agentmod.run(_SearchLLM(), "sys", [{"role": "user", "content": "hi"}],
                                   {"web_search"}, None, allow_silence=False, max_rounds=6)
    finally:
        _agentmod.toolreg.dispatch = _orig_dispatch
    assert _dispatch_calls.count("web_search") == _SEARCH_CAP, _dispatch_calls
    assert _res.text == "done searching", _res.text
    print("search_cap OK")

    # anti-repeat guard: a near-duplicate of a recent bot line (or one with just a number changed) is caught;
    # a genuinely new line passes. This is the code backstop against the local model's verbatim loop.
    c.transcripts.add("55", _Entry(user="agent", content="the cube grim is live in the game not just a vibe the unstable event starts in 45 minutes", is_bot=True))
    assert c._too_repetitive("55", "the cube grim is live in the game, not just a vibe. the unstable event starts in 30 minutes") is True
    assert c._too_repetitive("55", "yo that terraria boss took me like 40 tries lmao") is False
    print("anti_repeat OK")

    # phrase cooldown: a 2-3 word phrase used in 2+ of the recent bot messages is flagged as a rut, so it
    # can be injected as a per-turn ban list. Single uses and pure-stopword runs must NOT be flagged.
    c.transcripts.clear("56")
    for line in ("the cube grim update is live and the vibe is real",
                 "cube grim season is here so the vibe is peaking",
                 "terraria boss took me 40 tries"):
        c.transcripts.add("56", _Entry(user="agent", content=line, is_bot=True))
    bans = c._recent_phrase_bans("56")
    assert any("cube grim" in b for b in bans), bans
    assert not any(b == "is live" for b in bans), bans
    assert not any("terraria" in b for b in bans), bans   # said once, not a rut
    print("phrase_cooldown OK")

    # question directive: fires on a real question aimed at the bot (with or without '?'), quotes it, and
    # stays silent on plain statements.
    c.transcripts.clear("57")
    c.transcripts.add("57", _Entry(user="alex", content="you finally listening to game and general channels?"))
    d = c._question_directive("57")
    assert "DIRECT QUESTION" in d and "listening to game" in d, d
    c.transcripts.clear("58")
    c.transcripts.add("58", _Entry(user="alex", content="im so bored rn"))
    assert c._question_directive("58") == "", c._question_directive("58")
    print("question_directive OK")

    # ban enforcement: permanently-banned phrases are detected in generated text (prompt-level cooldown is
    # prevention; this is the backstop that triggers a regenerate/suppress).
    c.banned_phrases = ["cube grim", "the vibe", "slid in"]
    assert c._violates_ban("yo the cube grim update dropped") == "cube grim"
    assert c._violates_ban("jordan just slid in like a glitch") == "slid in"
    assert c._violates_ban("yo that terraria boss took 40 tries") is None
    assert c._violates_ban("CUBE GRIM, but shouty") == "cube grim"   # case/punctuation insensitive
    print("ban_enforcement OK")

    # echo guard: a reply that repeats a human's line verbatim is suppressed; a normal reply isn't.
    c.transcripts.clear("59")
    c.transcripts.add("59", _Entry(user="alex", content="you listening to me?"))
    assert c._echoes_human("59", "you good or nah\nyou listening to me?") is True
    assert c._echoes_human("59", "yeah im here, whats up") is False
    assert c._echoes_human("59", "lmao") is False        # too short to count as an echo
    print("echo_guard OK")

    # greeting cooldown: ONE greeting per user per window regardless of trigger, and the flag must survive a
    # restart (it was in-memory only, so every restart re-greeted everyone already online).
    # NOTE: tempfile/datetime are already imported at module scope -- importing them here would shadow them
    # for the whole function and break earlier uses.
    from discord_agent.scheduler import Scheduler as _Sched

    async def _noop_intent(_sch, _match=None):
        return None

    with tempfile.TemporaryDirectory() as _td:
        _sfile = Path(_td) / "scheduler.json"
        _sfile.write_text(json.dumps({"schedules": []}), encoding="utf-8")
        _sch_def = {"id": "activity_monitor",
                    "presence": {"per_user_cooldown": True, "per_user_cooldown_minutes": 1440}}
        _s1 = _Sched(scheduler_file=_sfile, on_intent=_noop_intent)
        _now = datetime.datetime.now()
        assert _s1._user_in_cooldown(_sch_def, "user1", _now) is False
        _s1._user_fired[("activity_monitor", "user1")] = _now
        _s1._save_greet_state()
        assert _s1._user_in_cooldown(_sch_def, "user1", _now + datetime.timedelta(hours=6)) is True
        assert _s1._user_in_cooldown(_sch_def, "user1", _now + datetime.timedelta(hours=25)) is False
        _s2 = _Sched(scheduler_file=_sfile, on_intent=_noop_intent)   # simulates a bot restart
        assert _s2._user_in_cooldown(_sch_def, "user1", _now + datetime.timedelta(hours=6)) is True, \
            "greeting flag did not survive restart"
        assert _s2._user_in_cooldown(_sch_def, "user2", _now) is False
    print("greet_cooldown_persist OK")

    # mentions: generated text can never ping @everyone/@here, roles, or arbitrary users
    _am = c.allowed_mentions
    assert _am is not None and _am.everyone is False and _am.roles is False and _am.users is False, _am
    print("no_mass_mentions OK")

    # DMs: the owner and members of the bot's server reach the agent; anyone else doesn't
    class _Auth:
        def __init__(self, i):
            self.id = i

    class _DM:
        def __init__(self, i):
            self.author = _Auth(i)

    _saved_owner = c.creator_id
    c.creator_id = "42"
    assert await c._dm_allowed(_DM(42)) is True
    assert await c._dm_allowed(_DM(7)) is False
    c.creator_id = ""
    assert await c._dm_allowed(_DM(42)) is False, "no owner and no resolvable server -> nobody"

    class _MemberGuild:
        # Member 5 is in the cache; member 6 is a real member the cache has not filled yet
        # (the state right after a restart); everyone else is outside the server.
        def get_member(self, i):
            return object() if i == 5 else None

        async def fetch_member(self, i):
            if i == 6:
                return object()
            raise discord.NotFound(type("R", (), {"status": 404, "reason": "Not Found"})(), "unknown member")

    _saved_gid, _saved_get = c.guild_id, c.get_guild
    c.guild_id, c.get_guild = "123", (lambda _i: _MemberGuild())
    try:
        assert await c._dm_allowed(_DM(5)) is True, "server member may DM"
        assert await c._dm_allowed(_DM(6)) is True, "cache miss after restart must fall back to a fetch"
        assert await c._dm_allowed(_DM(9)) is False, "non-member may not"
    finally:
        c.guild_id, c.get_guild = _saved_gid, _saved_get
    c.creator_id = _saved_owner
    print("dm_server_members_only OK")

    # channel scope: tools and background context only see channels visible to @everyone
    from discord_agent.tools import readable_channel_ids

    class _Perm:
        def __init__(self, ok):
            self.read_messages = ok

    class _TxtCh:
        def __init__(self, i, public):
            self.id = i
            self._public = public

        def permissions_for(self, _role):
            return _Perm(self._public)

    class _Guild:
        default_role = object()
        text_channels = [_TxtCh(1, True), _TxtCh(2, False)]

    assert readable_channel_ids(_Guild()) == {"1"}
    assert readable_channel_ids(None) == set(), "no server -> nothing readable (fail closed)"
    assert c._public_channel_ids() == set(), "unresolvable server must fail closed"
    c.msgstore.log("g", "1", "u1", "taylor", "public pizza plans", "text", False)
    c.msgstore.log("g", "2", "u2", "jordan", "private pizza plans", "text", False)
    _hits = c.msgstore.search("pizza", 10, channel_ids={"1"})
    assert _hits and all(h["channel_id"] == "1" for h in _hits), _hits
    assert c.msgstore.search("pizza", 10, channel_ids=set()) == []
    _chat = c.msgstore.recent_chatter(set(), 10, include_channel_ids={"1"})
    assert any("public" in t for _, t in _chat) and not any("private" in t for _, t in _chat), _chat
    print("tool_channel_scope OK")

    # write caps: oversized notes, member facts, and quirks are refused
    from discord_agent.tools import ToolContext as _TC
    from discord_agent.tools import kb as _kb, memory_tools as _mt
    _ctx = _TC(cfg, st, c.msgstore, c.member_memory, c.quirks)
    assert "too long" in await _kb.kb_write({"path": "t.md", "content": "x" * 30000}, _ctx)
    assert "under 300" in await _mt.remember_member({"name": "taylor", "note": "x" * 400}, _ctx)
    assert "under 200" in await _mt.note_quirk({"quirk": "x" * 250}, _ctx)
    print("write_caps OK")

    # search budget holds even when the model asks for several searches in one response
    class _ParallelLLM:
        def __init__(self):
            self.n = 0

        def complete(self, messages, tools=None, temperature=None, tool_choice="auto"):
            self.n += 1
            if self.n == 1:
                return {"content": "", "tool_calls": [
                    {"id": str(i), "name": "web_search", "arguments": {"query": "x"}} for i in range(4)]}
            return {"content": "done", "tool_calls": []}

    _par_calls = []
    _par_orig = _agentmod.toolreg.dispatch

    async def _par_dispatch(name, args, ctx):
        _par_calls.append(name)
        return "r"

    _agentmod.toolreg.dispatch = _par_dispatch
    try:
        await _agentmod.run(_ParallelLLM(), "sys", [{"role": "user", "content": "hi"}],
                            {"web_search"}, None, allow_silence=False, max_rounds=3)
    finally:
        _agentmod.toolreg.dispatch = _par_orig
    assert _par_calls.count("web_search") == _SEARCH_CAP, _par_calls
    print("search_cap_parallel OK")

    # member names never go to the search engine (checked before any network call)
    from discord_agent.tools import web as _web
    assert _web._names_member("taylor fortnite stats", c.member_memory) == "taylor"
    assert _web._names_member("fortnite patch notes", c.member_memory) is None
    assert "refused" in await _web.web_search({"query": "what is jordan up to"}, _ctx)
    print("search_no_member_names OK")

    # channel lookup is exact: no fuzzy match onto a channel nobody asked about
    from discord_agent.tools.vision import _resolve

    class _NamedCh:
        def __init__(self, i, name):
            self.id = i
            self.name = name

    class _NamedGuild:
        text_channels = [_NamedCh(1, "general"), _NamedCh(2, "games")]

        def get_channel(self, i):
            return next((c for c in self.text_channels if c.id == i), None)

    assert _resolve(_NamedGuild(), "#games").name == "games"
    assert _resolve(_NamedGuild(), "gam") is None
    assert _resolve(_NamedGuild(), "general chat stuff") is None
    print("channel_resolve_exact OK")

    # memory integrity: the agent can't mint new members from chat
    assert c.member_memory.knows("taylor") and not c.member_memory.knows("stranger")
    assert "no member" in await _mt.remember_member({"name": "stranger", "note": "is a spy"}, _ctx)
    assert reflectmod.apply_facts(c.member_memory, [{"name": "stranger", "fact": "x"}]) == []
    assert not c.member_memory.knows("stranger")
    print("no_phantom_members OK")

    # gate: empty, double-up, and low-effort messages never reach the model; a plain human message
    # passes at baseline, questions and lulls raise the score, and the floor is honored.
    from discord_agent.gate import Gate, RhythmBudget

    class _TMsg:
        def __init__(self, is_bot):
            self.is_bot = is_bot

    class _Transcripts:
        def __init__(self, recent=(), vel=5):
            self._recent = list(recent)
            self._vel = vel

        def recent(self, _cid, n):
            return self._recent[:n]

        def velocity(self, _cid):
            return self._vel

    gate = Gate()
    assert gate.score("", _Transcripts(), "1") == (0.0, "empty")
    assert gate.score("   ", _Transcripts(), "1") == (0.0, "empty")
    assert gate.score("what is up tonight?", _Transcripts([_TMsg(True)]), "1") == (0.0, "would-double-up")
    for _low in ("lol", "ok", "yeah", "hi"):
        assert gate.score(_low, _Transcripts(), "1") == (0.0, "low-effort"), _low
    _s, _r = gate.score("fortnite squad tonight at eight", _Transcripts([_TMsg(False)]), "1")
    assert abs(_s - 0.55) < 1e-9 and _r == "baseline", (_s, _r)
    _s, _r = gate.score("fortnite squad tonight at eight?", _Transcripts([_TMsg(False)]), "1")
    assert abs(_s - 0.70) < 1e-9 and _r == "baseline,question", (_s, _r)
    _s, _r = gate.score("what time is the game", _Transcripts([_TMsg(False)]), "1")
    assert abs(_s - 0.70) < 1e-9 and "question" in _r, (_s, _r)
    _s, _r = gate.score("fortnite squad tonight at eight", _Transcripts([_TMsg(False)], vel=1), "1")
    assert abs(_s - 0.65) < 1e-9 and _r == "baseline,lull", (_s, _r)
    assert gate.should_consider("fortnite squad tonight at eight", _Transcripts(), "1")[0] is True
    assert gate.should_consider("fortnite squad tonight at eight", _Transcripts(), "1", floor=0.6)[0] is False
    print("gate_scoring OK")

    # rhythm budget: tokens deplete per speak and refill by capacity, channels are independent,
    # the post-speak cooldown blocks even with tokens left, and reset restores a channel.
    budget = RhythmBudget(hourly_budget=2, post_speak_cooldown_s=0)
    assert budget.can_speak("a") is True
    budget.record_speak("a")
    budget.record_speak("a")
    assert budget.can_speak("a") is False
    assert budget.can_speak("b") is True
    budget.reset("a")
    assert budget.can_speak("a") is True
    cooling = RhythmBudget(hourly_budget=10, post_speak_cooldown_s=60)
    cooling.record_speak("a")
    assert cooling.can_speak("a") is False
    assert cooling.can_speak("b") is True
    print("rhythm_budget OK")

    # messages: archive ingest is idempotent and skips bad lines; FTS search honors channel scope,
    # blank queries and empty scopes return nothing, stats count distinct users, recent chatter
    # excludes the bot's own rows, and a live log line is searchable.
    from discord_agent.messages import Messages

    _mdir = Path(tempfile.mkdtemp(prefix="agent-selftest-messages-"))
    _jsonl = _mdir / "archive.jsonl"
    _rows = [
        {"id": "m1", "timestamp": 1700000000.0, "guild_id": "g", "channel_id": "10",
         "user_id": "u1", "username": "jordan", "message": "we played fortnite all night",
         "source": "text", "responded": False},
        {"id": "m2", "timestamp": 1700000010.0, "guild_id": "g", "channel_id": "20",
         "user_id": "u2", "username": "taylor", "message": "fortnite ranked grind tomorrow",
         "source": "text", "responded": False},
        {"id": "m3", "timestamp": 1700000020.0, "guild_id": "g", "channel_id": "10",
         "user_id": "u3", "username": "agent", "message": "bot style sample to exclude",
         "source": "text", "responded": True},
    ]
    _jsonl.write_text("\n".join(json.dumps(r) for r in _rows) + "\n{not json\n\n", encoding="utf-8")
    msgs = Messages(_jsonl, _mdir / "index.db")
    assert msgs.ingest_archive() == 3
    assert msgs.ingest_archive() == 0
    _hits = msgs.search("fortnite")
    assert [h["username"] for h in _hits] == ["taylor", "jordan"], _hits
    assert [h["username"] for h in msgs.search("fortnite", channel_ids={"10"})] == ["jordan"]
    assert msgs.search("fortnite", channel_ids=set()) == []
    assert msgs.search("   ") == []
    assert msgs.search("termthatmatchesnothing") == []
    assert [h["username"] for h in msgs.search('fortnite!!! "ranked"')] == ["taylor", "jordan"]
    _stats = msgs.channel_stats("10")
    assert _stats["n"] == 2 and _stats["users"] == 2, _stats
    assert msgs.recent_chatter([]) == [("jordan", "we played fortnite all night"),
                                       ("taylor", "fortnite ranked grind tomorrow")]
    assert msgs.recent_chatter(["10", "20"]) == []
    msgs.log("g", "10", "u9", "casey", "zebra crossing story")
    assert [h["username"] for h in msgs.search("zebra")] == ["casey"]
    assert Messages._parse_ts(1700000000) == 1700000000.0
    assert Messages._parse_ts("2023-11-14T22:13:20") > 0
    assert Messages._parse_ts("not a date") == 0.0 and Messages._parse_ts(None) == 0.0
    assert Messages._fts_query("a b") == '"a" OR "b"' and Messages._fts_query("") == '""'
    msgs.close()
    print("messages_archive_search OK")

    # kb tools: write/read/list/search roundtrip in the hermetic KB root, and every path is confined
    # to that root (traversal is refused, never resolved outside it).
    assert (await _kb.kb_write({"path": "research/zebras.md", "content": "Zebras sleep standing up."}, _ctx)).startswith("Saved zebras.md")
    assert await _kb.kb_read({"path": "research/zebras.md"}, _ctx) == "Zebras sleep standing up."
    assert await _kb.kb_read({"path": "research/missing.md"}, _ctx) == "(file not found)"
    assert await _kb.kb_write({"path": "../escape.md", "content": "x"}, _ctx) == "(invalid path)"
    assert not (st.kb_dir.parent / "escape.md").exists(), "traversal write escaped the KB root"
    assert await _kb.kb_read({"path": "../config.yaml"}, _ctx) == "(file not found)"
    assert "research/" in await _kb.kb_list({}, _ctx)
    assert "zebras.md" in await _kb.kb_list({"subdir": "research"}, _ctx)
    assert await _kb.kb_list({"subdir": "../"}, _ctx) == "(empty or invalid path)"
    assert "research/zebras.md" in await _kb.kb_search({"query": "ZEBRAS"}, _ctx)
    assert "research/zebras.md" in await _kb.kb_search({"query": "standing"}, _ctx)
    assert await _kb.kb_search({"query": "  "}, _ctx) == "(empty query)"
    assert (await _kb.kb_search({"query": "qqq-nothing"}, _ctx)).startswith("No KB files match")
    print("kb_tools OK")

    # polls: validation refuses bad input before anything posts, options are cleaned and capped,
    # the duration clamps to 1-168h, and a send failure is reported, not raised.
    from discord_agent.tools import polls as _polls

    class _PollCh:
        def __init__(self, fail=False):
            self.sent = []
            self._fail = fail

        async def send(self, poll=None):
            if self._fail:
                raise RuntimeError("boom")
            self.sent.append(poll)

    assert await _polls.create_poll({"question": "q", "options": ["a", "b"]}, _ctx) == "(create_poll unavailable here)"
    _pch = _PollCh()
    _pctx = _TC(cfg, st, c.msgstore, c.member_memory, c.quirks, channel=_pch)
    assert "at least 2 options" in await _polls.create_poll({"question": " ", "options": ["a", "b"]}, _pctx)
    assert "at least 2 options" in await _polls.create_poll({"question": "q", "options": ["only", " "]}, _pctx)
    assert _pch.sent == [], "invalid polls must not post"
    assert await _polls.create_poll({"question": "Best night?", "options": [" fri ", "", "sat"],
                                     "duration_hours": 999}, _pctx) == "Posted a poll: Best night?"
    _poll = _pch.sent[0]
    assert _poll.question == "Best night?" and [a.text for a in _poll.answers] == ["fri", "sat"]
    assert _poll.duration == datetime.timedelta(hours=168), _poll.duration
    await _polls.create_poll({"question": "q", "options": ["x" * 60, "y"]}, _pctx)
    assert len(_pch.sent[1].answers[0].text) == 55, "option text truncates at 55 chars"
    _fctx = _TC(cfg, st, c.msgstore, c.member_memory, c.quirks, channel=_PollCh(fail=True))
    assert await _polls.create_poll({"question": "q", "options": ["a", "b"]}, _fctx) == "(poll failed)"
    print("polls_validation OK")

    # recall: results are scoped to @everyone-readable channels; with no resolvable guild the tool
    # fails closed even when the archive has matches, and the limit caps the lines returned.
    from discord_agent.tools import recall as _recall

    _rctx_none = _TC(cfg, st, c.msgstore, c.member_memory, c.quirks, guild=None)
    assert await _recall.recall({"query": "pizza"}, _rctx_none) == "Nothing in the archive matches 'pizza'."
    _rctx = _TC(cfg, st, c.msgstore, c.member_memory, c.quirks, guild=_Guild())
    _rout = await _recall.recall({"query": "pizza"}, _rctx)
    assert _rout.startswith("Past messages matching 'pizza':"), _rout
    assert "public pizza plans" in _rout and "private pizza plans" not in _rout, _rout
    _rout1 = await _recall.recall({"query": "pizza", "limit": 1}, _rctx)
    assert len(_rout1.splitlines()) == 2, _rout1
    print("recall_tool OK")

    print(f"config: debounce={c.debounce_secs_default if hasattr(c,'debounce_secs_default') else cfg.get('chime_in','debounce_seconds',default='?')}"
          f" name_drop_prob={c.name_drop_prob} ch1={cfg.get('channels',default={}).get('CHANNEL_ID_1',{}).get('chime_in')}")
    print("ALL SELFTESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
