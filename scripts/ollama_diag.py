#!/usr/bin/env python3
"""Diagnose local Ollama latency on the model host: where does the time actually go?

Measures, per trial:
  - load_duration       (model load, 0 if already resident)
  - prompt_eval         (PREFILL: tokens + tok/s)  <-- usually the hidden cost
  - eval                (GENERATION: tokens + tok/s)
  - TTFT                (time to first token, via streaming)
  - inter-token latency (p50/p95 gaps between streamed tokens)
Sweeps prompt size so prefill scaling is visible (DiscordAgent sends a big
system prompt + transcript, not a 30-token toy prompt).

Also samples CPU/RAM (and GPU via nvidia-smi) during generation, and reports
hardware facts (RAM speed, channels, cores) to compute the theoretical
memory-bandwidth ceiling -- the real limit for local LLM generation.

Run on the model host:  python ollama_diag.py
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HOST = "http://127.0.0.1:11434"
MODEL = "qwen3:30b-a3b-instruct-2507-q8_0"

# Active params for this MoE (30.5B total, ~3.3B active/token) at Q8 => ~1 byte/param.
ACTIVE_PARAMS_B = 3.3
BYTES_PER_PARAM = 1.0  # Q8_0


def post(path: str, payload: dict, timeout: int = 900):
    req = urllib.request.Request(
        HOST + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get(path: str, timeout: int = 30):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.loads(r.read())


def ps_out(cmd: str) -> str:
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(failed: {e})"


# ---------------------------------------------------------------- hardware


def hardware() -> dict:
    """Facts that set the performance ceiling. RAM speed x channels = bandwidth."""
    raw = ps_out(
        "$m=Get-CimInstance Win32_PhysicalMemory; "
        "$cs=Get-CimInstance Win32_ComputerSystem; $c=Get-CimInstance Win32_Processor; "
        "[pscustomobject]@{ dimms=($m|Measure-Object).Count; "
        "speeds=(($m|ForEach-Object{$_.ConfiguredClockSpeed}) -join ','); "
        "capacities=(($m|ForEach-Object{[math]::Round($_.Capacity/1GB)}) -join ','); "
        "totalGB=[math]::Round($cs.TotalPhysicalMemory/1GB,1); "
        "cores=$c.NumberOfCores; threads=$c.NumberOfLogicalProcessors; "
        "cpu=$c.Name } | ConvertTo-Json -Compress"
    )
    try:
        hw = json.loads(raw)
    except Exception:  # noqa: BLE001
        hw = {"raw": raw}

    gpu = ps_out(
        "try { nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits } catch { 'n/a' }"
    )
    hw["gpu"] = gpu
    return hw


def theoretical_bandwidth(hw: dict) -> float | None:
    """Peak DRAM bandwidth GB/s = speed(MT/s) * 8 bytes * channels."""
    try:
        speeds = [int(s) for s in str(hw.get("speeds", "")).split(",") if s.strip()]
        if not speeds:
            return None
        speed = max(speeds)
        dimms = int(hw.get("dimms") or len(speeds))
        channels = 2 if dimms >= 2 else 1  # i5-10600 = dual-channel max
        return speed * 8 * channels / 1000.0
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- sampling


class Sampler(threading.Thread):
    """Poll CPU/RAM/GPU while a generation runs."""

    def __init__(self, interval: float = 1.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cpu: list[float] = []
        self.gpu_util: list[float] = []
        self.free_gb: list[float] = []

    def run(self) -> None:
        while not self.stop_flag.is_set():
            out = ps_out(
                "$c=(Get-CimInstance Win32_Processor).LoadPercentage; "
                "$f=[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1); "
                "$g=try{ (nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits) }catch{ '' }; "
                "\"$c|$f|$g\""
            )
            parts = out.split("|")
            try:
                if parts[0].strip():
                    self.cpu.append(float(parts[0]))
                if len(parts) > 1 and parts[1].strip():
                    self.free_gb.append(float(parts[1]))
                if len(parts) > 2 and parts[2].strip():
                    self.gpu_util.append(float(parts[2].splitlines()[0]))
            except ValueError:
                pass
            self.stop_flag.wait(self.interval)

    def summary(self) -> str:
        def agg(vals, unit):
            if not vals:
                return "n/a"
            return f"avg {statistics.mean(vals):.0f}{unit} peak {max(vals):.0f}{unit}"
        return (f"cpu {agg(self.cpu, '%')} | gpu {agg(self.gpu_util, '%')} | "
                f"free_ram {agg(self.free_gb, 'GB')}")


# ---------------------------------------------------------------- one trial


def trial(label: str, prompt: str, num_predict: int, sample: bool = True) -> dict:
    """Stream one generation; capture TTFT, token gaps, and server-side counters."""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": num_predict},
    }
    sampler = Sampler() if sample else None
    if sampler:
        sampler.start()

    t0 = time.perf_counter()
    ttft = None
    gaps: list[float] = []
    last = None
    final: dict = {}
    try:
        resp = post("/api/generate", payload)
        for raw in resp:
            if not raw.strip():
                continue
            obj = json.loads(raw)
            now = time.perf_counter()
            if obj.get("response"):
                if ttft is None:
                    ttft = now - t0
                else:
                    gaps.append(now - last)
                last = now
            if obj.get("done"):
                final = obj
    finally:
        if sampler:
            sampler.stop_flag.set()
            sampler.join(timeout=5)

    wall = time.perf_counter() - t0

    def sec(key):
        return (final.get(key) or 0) / 1e9

    pe_count, pe_sec = final.get("prompt_eval_count") or 0, sec("prompt_eval_duration")
    ev_count, ev_sec = final.get("eval_count") or 0, sec("eval_duration")

    res = {
        "label": label,
        "prompt_tokens": pe_count,
        "gen_tokens": ev_count,
        "load_s": round(sec("load_duration"), 2),
        "prefill_s": round(pe_sec, 2),
        "prefill_tps": round(pe_count / pe_sec, 1) if pe_sec else 0,
        "gen_s": round(ev_sec, 2),
        "gen_tps": round(ev_count / ev_sec, 2) if ev_sec else 0,
        "ttft_s": round(ttft, 2) if ttft else None,
        "wall_s": round(wall, 2),
        "gap_p50_ms": round(statistics.median(gaps) * 1000, 1) if gaps else None,
        "gap_p95_ms": (round(sorted(gaps)[int(len(gaps) * 0.95)] * 1000, 1)
                       if len(gaps) >= 20 else None),
        "sys": sampler.summary() if sampler else "",
    }
    return res


def filler(approx_tokens: int) -> str:
    """~4 chars/token of realistic chat-log-ish text."""
    line = ("user123: yo did you see that new patch they dropped last night it broke "
            "everything again lmao and now the lobby wont even load properly\n")
    reps = max(1, int(approx_tokens / 28))
    return line * reps


def main() -> int:
    print("=" * 74)
    print("OLLAMA LATENCY DIAGNOSTIC")
    print("=" * 74)

    hw = hardware()
    print("\n--- HARDWARE ---")
    for k in ("cpu", "cores", "threads", "totalGB", "dimms", "speeds", "capacities", "gpu"):
        if k in hw:
            print(f"  {k:11}: {hw[k]}")
    bw = theoretical_bandwidth(hw)
    if bw:
        print(f"  {'peak_bw':11}: ~{bw:.1f} GB/s theoretical DRAM bandwidth")

    try:
        loaded = get("/api/ps").get("models", [])
        for m in loaded:
            vram = (m.get("size_vram") or 0) / 1e9
            total = (m.get("size") or 0) / 1e9
            print(f"\n--- RESIDENT: {m.get('name')} ---")
            print(f"  total {total:.1f} GB | in VRAM {vram:.1f} GB "
                  f"({100 * vram / total:.0f}% GPU-offloaded) | ctx {m.get('context_length')}")
    except Exception as e:  # noqa: BLE001
        print(f"  (/api/ps failed: {e})")

    trials = [
        ("tiny prompt / short gen", filler(0) + "Say hi in five words.", 24),
        ("small prompt (~250 tok)", filler(250) + "\nReply with one short sentence.", 40),
        ("medium prompt (~1000 tok)", filler(1000) + "\nReply with one short sentence.", 40),
        ("large prompt (~2500 tok)", filler(2500) + "\nReply with one short sentence.", 40),
        ("tiny prompt / long gen", "Write a detailed paragraph about winter.", 200),
    ]

    print("\n--- TRIALS ---")
    rows = []
    for label, prompt, npred in trials:
        print(f"\n[{label}] running...")
        try:
            r = trial(label, prompt, npred)
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            continue
        rows.append(r)
        print(f"  prompt_tokens={r['prompt_tokens']}  gen_tokens={r['gen_tokens']}")
        print(f"  TTFT={r['ttft_s']}s   (prefill {r['prefill_s']}s @ {r['prefill_tps']} tok/s)")
        print(f"  gen={r['gen_s']}s @ {r['gen_tps']} tok/s   wall={r['wall_s']}s")
        if r["gap_p50_ms"]:
            print(f"  inter-token: p50 {r['gap_p50_ms']}ms  p95 {r['gap_p95_ms']}ms")
        print(f"  {r['sys']}")

    print("\n" + "=" * 74)
    print("ANALYSIS")
    print("=" * 74)
    gen_rates = [r["gen_tps"] for r in rows if r["gen_tps"]]
    if gen_rates:
        best = max(gen_rates)
        bytes_per_tok = ACTIVE_PARAMS_B * BYTES_PER_PARAM
        eff = best * bytes_per_tok
        print(f"  best generation: {best:.2f} tok/s")
        print(f"  ~{bytes_per_tok:.1f} GB must move per token (MoE active params @ Q8)")
        print(f"  => effective memory bandwidth ~{eff:.1f} GB/s")
        if bw:
            print(f"  => {100 * eff / bw:.0f}% of the {bw:.1f} GB/s theoretical peak")
            print("     (>60% means you are memory-bandwidth-bound: the CPU is waiting on RAM,")
            print("      so more cores or GPU tweaks won't help -- only a smaller quant,")
            print("      fewer bytes/token, or faster memory will.)")
    pf = [(r["prompt_tokens"], r["prefill_tps"]) for r in rows if r["prompt_tokens"] > 100]
    if pf:
        print("\n  prefill (time-to-first-token driver):")
        for toks, tps in pf:
            print(f"    {toks:>5} prompt tokens @ {tps:>6.1f} tok/s "
                  f"=> {toks / tps:>5.1f}s before the first word")
        print("  Real replies carry a big system prompt + transcript, so THIS is usually")
        print("  what 'slow to respond' actually is -- and an agent loop pays it once PER")
        print("  LLM call (tool round-trips multiply it).")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
