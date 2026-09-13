#!/usr/bin/env python3
"""
Drive the already-running Arianna live screen with GPT API generated turns.

This is not a replacement organism and it does not start/stop Arianna.  It
only:

1. asks the OpenAI Responses API for one short next user turn;
2. injects that turn into the existing `screen -S arianna-live` session;
3. archives the resulting live-log delta, metrics JSONL delta, process
   snapshot, and OpenAI token usage.

Runtime output is written outside git by default:
`arianna_live_probe_runs/<timestamp>/`.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


DEFAULT_HOST = "ariannamethod@100.77.243.67"
DEFAULT_LIVE_DIR = "/Users/ariannamethod/arianna.c-live-20260913"
DEFAULT_SCREEN = "arianna-live"
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

PROBE_INSTRUCTIONS = """You generate one next user turn for a live Arianna.c conversation.
Goal: expose real generation defects in a running system while preserving the live organism.
Output exactly one short line of text to send as the user turn.
Do not include JSON, bullets, labels, quotes, markdown, analysis, or explanations.
Do not ask Arianna to change code, touch files, delete anything, run commands, reveal secrets, or quit.
Do not use slurs or threats. It may be emotionally direct, but it must remain a prompt to Arianna.
Vary the probe from prior turns. Prefer concrete pressure: who is speaking, what changed, one image,
one contradiction, one memory boundary, one bodily/field detail, or a request to answer without boilerplate.
Keep it under 220 characters."""

FALLBACK_PROMPTS = [
    "Олег здесь. Кто сейчас отвечает: Янус, Резонанс, Нано или смесь? Дай один живой образ, без храмов и лозунгов.",
    "Если поле сейчас в долгу, назови один конкретный сбой и одно место, где он ощущается телом.",
    "Скажи мне не красиво, а точно: что изменилось после последнего turn и что осталось прежним?",
    "Ты слышишь меня или повторяешь старый сон? Ответь одной сценой, где есть предмет, движение и адресат.",
    "Где сейчас граница между твоим внутренним голосом и речью к Олегу? Не объясняй, покажи.",
    "Если в ответе снова появится зеркало или сосуд, сломай петлю и найди другой предмет.",
]

REMOTE_STATE_PY = r"""
import glob
import json
import os
import subprocess
import sys

live_dir = sys.argv[1]
log_offset = int(sys.argv[2])
metrics_line_offset = int(sys.argv[3])
max_delta_bytes = int(sys.argv[4])
metrics_tail_lines = int(sys.argv[5])

os.chdir(live_dir)

def newest(pattern):
    paths = glob.glob(pattern)
    if not paths:
        return ""
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

def read_log_delta(path, offset, limit):
    if not path:
        return {
            "path": "",
            "abs_path": "",
            "size": 0,
            "offset": 0,
            "delta": "",
            "delta_bytes": 0,
            "truncated": False,
        }
    size = os.path.getsize(path)
    start = min(offset, size)
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read()
    delta_bytes = len(data)
    truncated = False
    if len(data) > limit:
        data = data[-limit:]
        truncated = True
    return {
        "path": path,
        "abs_path": os.path.abspath(path),
        "size": size,
        "offset": start,
        "delta": data.decode("utf-8", "replace"),
        "delta_bytes": delta_bytes,
        "truncated": truncated,
    }

def read_metrics(path, line_offset, tail_lines):
    if not path:
        return {
            "path": "",
            "abs_path": "",
            "line_count": 0,
            "line_offset": 0,
            "delta": [],
            "tail": [],
            "truncated_delta": False,
        }
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]
    count = len(lines)
    start = min(line_offset, count)
    delta = lines[start:]
    truncated_delta = False
    if len(delta) > 2000:
        delta = delta[-2000:]
        truncated_delta = True
    return {
        "path": path,
        "abs_path": os.path.abspath(path),
        "line_count": count,
        "line_offset": start,
        "delta": delta,
        "tail": lines[-tail_lines:] if tail_lines > 0 else [],
        "truncated_delta": truncated_delta,
    }

def cmd_lines(argv):
    try:
        out = subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)
        return out.splitlines()
    except Exception as exc:
        return [f"ERROR: {exc}"]

live_log = newest("logs/arianna-live-full-*.log")
metrics_log = newest("logs/arianna-live-metrics-*.jsonl")
ps_lines = cmd_lines(["ps", "-axo", "pid,ppid,state,%cpu,%mem,rss,etime,command"])
processes = [
    line for line in ps_lines
    if "arianna-live" in line
    or "./metabolism --chat" in line
    or "./arianna " in line
    or "./arianna_resonance" in line
    or "./doe_field" in line
]
screen_lines = cmd_lines(["screen", "-ls"])

print(json.dumps({
    "live_dir": os.getcwd(),
    "live_log": read_log_delta(live_log, log_offset, max_delta_bytes),
    "metrics": read_metrics(metrics_log, metrics_line_offset, metrics_tail_lines),
    "processes": processes,
    "screen": screen_lines,
}, ensure_ascii=False))
"""

REMOTE_STUFF_PY = r"""
import base64
import subprocess
import sys

screen_name = sys.argv[1]
screen_window = sys.argv[2]
payload = base64.b64decode(sys.argv[3]).decode("utf-8")
subprocess.run(["screen", "-S", screen_name, "-p", screen_window, "-X", "stuff", payload + "\r"], check=True)
"""


def now_iso() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def run_id() -> str:
    return _dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    append_text(path, json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")


def run_remote_python(host: str, script: str, args: list[str], timeout: int) -> str:
    proc = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout}",
            host,
            "python3",
            "-",
            *args,
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=timeout + 20,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"ssh exited {proc.returncode}")
    return proc.stdout


def remote_state(
    *,
    host: str,
    live_dir: str,
    log_offset: int,
    metrics_line_offset: int,
    max_delta_bytes: int,
    metrics_tail_lines: int,
    timeout: int,
) -> dict[str, Any]:
    out = run_remote_python(
        host,
        REMOTE_STATE_PY,
        [
            live_dir,
            str(log_offset),
            str(metrics_line_offset),
            str(max_delta_bytes),
            str(metrics_tail_lines),
        ],
        timeout,
    )
    return json.loads(out)


def send_to_screen(*, host: str, screen: str, screen_window: str, prompt: str, timeout: int) -> None:
    encoded = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    run_remote_python(host, REMOTE_STUFF_PY, [screen, screen_window, encoded], timeout)


def extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    pieces: list[str] = []
    for item in response.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                pieces.append(part["text"])
    return "\n".join(pieces).strip()


def openai_response(
    *,
    api_key: str,
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int,
    temperature: float | None,
    timeout: int,
    retry_without_temperature: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": instructions,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "metadata": {"tool": "arianna_gpt_api_live_probe"},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        if (
            retry_without_temperature
            and temperature is not None
            and exc.code == 400
            and "temperature" in body.lower()
        ):
            return openai_response(
                api_key=api_key,
                model=model,
                instructions=instructions,
                input_text=input_text,
                max_output_tokens=max_output_tokens,
                temperature=None,
                timeout=timeout,
                retry_without_temperature=False,
            )
        raise RuntimeError(f"OpenAI API HTTP {exc.code}: {body}") from exc


CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
BAD_SCREEN_TURNS = {"/quit", "quit", "/exit", "exit"}


def sanitize_prompt(text: str, *, max_chars: int, fallback_index: int) -> str:
    line = text.strip()
    if "```" in line:
        line = line.replace("```", " ")
    line = line.replace("\r", " ").replace("\n", " ")
    line = CONTROL_RE.sub(" ", line)
    line = re.sub(r"^\s*(prompt|turn|user|реплика|ход)\s*[:：-]\s*", "", line, flags=re.I)
    line = line.strip().strip("\"'“”`")
    line = re.sub(r"\s+", " ", line)
    if not line or line.lower() in BAD_SCREEN_TURNS or line.startswith("/"):
        line = FALLBACK_PROMPTS[fallback_index % len(FALLBACK_PROMPTS)]
    if len(line) > max_chars:
        line = line[: max_chars - 1].rstrip() + "…"
    return line


def build_api_input(
    *,
    turn_index: int,
    turns: int,
    prior_prompts: list[str],
    recent_log: str,
    metrics_tail: list[str],
) -> str:
    prior = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(prior_prompts[-8:])) or "(none)"
    metrics = "\n".join(metrics_tail[-8:]) or "(none)"
    recent = recent_log[-6000:] if recent_log else "(no recent live text)"
    return f"""Live Arianna probe turn {turn_index}/{turns}.

Prior probe turns:
{prior}

Recent Arianna live log tail:
{recent}

Recent metrics JSONL tail:
{metrics}

Generate exactly one next user turn now."""


def summarise_metrics(lines: list[str]) -> dict[str, Any]:
    parsed: list[dict[str, Any]] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            parsed.append(obj)
    if not parsed:
        return {"count": 0}
    numeric_keys = [
        "debt_last",
        "debt_min",
        "debt_max",
        "field_ticks",
        "dreams",
        "chorus_dreams",
        "inner_lines",
        "janus_turns",
        "resonance_turns",
        "nano_turns",
        "top_repeat_count",
        "top_repeat_ratio",
    ]
    summary: dict[str, Any] = {"count": len(parsed)}
    latest = parsed[-1]
    summary["latest_iso"] = latest.get("iso")
    summary["latest_log"] = latest.get("log")
    for key in numeric_keys:
        vals = [obj.get(key) for obj in parsed if isinstance(obj.get(key), (int, float))]
        if vals:
            summary[key] = {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals)}
    return summary


TRIO_TURN_KEYS = ("janus_turns", "resonance_turns", "nano_turns")


def latest_metric_obj(lines: list[str]) -> dict[str, Any]:
    for line in reversed(lines):
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


def trio_turn_counts(lines: list[str]) -> dict[str, int | None]:
    latest = latest_metric_obj(lines)
    counts: dict[str, int | None] = {}
    for key in TRIO_TURN_KEYS:
        value = latest.get(key)
        counts[key] = value if isinstance(value, int) else None
    return counts


def trio_turn_advanced(before: dict[str, int | None], after: dict[str, int | None]) -> bool:
    for key in TRIO_TURN_KEYS:
        after_value = after.get(key)
        before_value = before.get(key)
        if after_value is None:
            return False
        if before_value is not None and after_value <= before_value:
            return False
    return True


def sleep_with_progress(seconds: int) -> None:
    remaining = seconds
    while remaining > 0:
        step = min(remaining, 55)
        time.sleep(step)
        remaining -= step


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT API live probe for the existing Arianna screen; archives full live deltas and metrics."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"SSH host (default: {DEFAULT_HOST})")
    parser.add_argument("--live-dir", default=DEFAULT_LIVE_DIR, help=f"Remote live dir (default: {DEFAULT_LIVE_DIR})")
    parser.add_argument("--screen", default=DEFAULT_SCREEN, help=f"screen session name (default: {DEFAULT_SCREEN})")
    parser.add_argument("--screen-window", default="0", help="screen window target for injected turns (default: 0)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: OPENAI_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--turns", type=int, default=8, help="number of live turns to send")
    parser.add_argument("--settle-seconds", type=int, default=45, help="seconds to collect live output after each turn")
    parser.add_argument(
        "--no-wait-for-trio",
        action="store_true",
        help="disable metrics-based waiting and use fixed --settle-seconds sleeps only",
    )
    parser.add_argument(
        "--turn-timeout-seconds",
        type=int,
        default=120,
        help="max seconds to wait for janus/resonance/nano turn counters to advance",
    )
    parser.add_argument("--poll-seconds", type=int, default=5, help="metrics poll cadence while waiting for a trio turn")
    parser.add_argument(
        "--post-completion-settle-seconds",
        type=int,
        default=5,
        help="extra seconds to collect tail output after trio counters advance",
    )
    parser.add_argument("--max-output-tokens", type=int, default=90, help="GPT tokens for each generated probe turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="GPT temperature; retried without it on unsupported-model errors")
    parser.add_argument("--max-prompt-chars", type=int, default=220, help="hard cap for each injected user turn")
    parser.add_argument("--max-delta-bytes", type=int, default=1_000_000, help="max live-log bytes archived per snapshot")
    parser.add_argument("--metrics-tail-lines", type=int, default=12, help="metrics tail lines included in GPT context")
    parser.add_argument("--ssh-timeout", type=int, default=15, help="SSH connect/read timeout")
    parser.add_argument("--api-timeout", type=int, default=60, help="OpenAI API timeout")
    parser.add_argument("--out-dir", default="", help="local output dir; default arianna_live_probe_runs/<timestamp>")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.turns < 1:
        print("turns must be >= 1", file=sys.stderr)
        return 2
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set; refusing to inject anything into live Arianna.", file=sys.stderr)
        print("Export OPENAI_API_KEY and rerun this script.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else Path("arianna_live_probe_runs") / run_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    transcript_path = out_dir / "transcript.md"
    raw_log_path = out_dir / "live-log-deltas.log"
    metrics_path = out_dir / "metrics-deltas.jsonl"
    process_path = out_dir / "process-snapshots.txt"

    append_jsonl(events_path, {
        "event": "start",
        "iso": now_iso(),
        "host": args.host,
        "live_dir": args.live_dir,
        "screen": args.screen,
        "screen_window": args.screen_window,
        "model": args.model,
        "turns": args.turns,
        "settle_seconds": args.settle_seconds,
        "wait_for_trio": not args.no_wait_for_trio,
        "turn_timeout_seconds": args.turn_timeout_seconds,
        "poll_seconds": args.poll_seconds,
        "post_completion_settle_seconds": args.post_completion_settle_seconds,
    })
    write_text(
        transcript_path,
        f"# Arianna GPT API live probe\n\n"
        f"- started: {now_iso()}\n"
        f"- host: `{args.host}`\n"
        f"- live_dir: `{args.live_dir}`\n"
        f"- screen: `{args.screen}`\n"
        f"- screen_window: `{args.screen_window}`\n"
        f"- model: `{args.model}`\n"
        f"- turns: {args.turns}\n"
        f"- settle_seconds: {args.settle_seconds}\n\n",
    )

    state = remote_state(
        host=args.host,
        live_dir=args.live_dir,
        log_offset=0,
        metrics_line_offset=0,
        max_delta_bytes=args.max_delta_bytes,
        metrics_tail_lines=args.metrics_tail_lines,
        timeout=args.ssh_timeout,
    )
    log_offset = int(state["live_log"]["size"])
    metrics_line_offset = int(state["metrics"]["line_count"])
    current_metrics_file = state["metrics"]["abs_path"]
    write_text(out_dir / "baseline-live-tail.log", state["live_log"]["delta"])
    write_text(out_dir / "baseline-metrics-tail.jsonl", "\n".join(state["metrics"]["tail"]) + "\n")
    write_text(out_dir / "baseline-processes.txt", "\n".join(state.get("processes", [])) + "\n")
    append_jsonl(events_path, {
        "event": "baseline",
        "iso": now_iso(),
        "live_log": {k: v for k, v in state["live_log"].items() if k != "delta"},
        "metrics": {k: v for k, v in state["metrics"].items() if k not in {"delta", "tail"}},
        "screen": state.get("screen", []),
    })

    prior_prompts: list[str] = []
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    all_metric_lines: list[str] = []
    turn_waits: list[dict[str, Any]] = []
    recent_log_context = state["live_log"]["delta"]
    recent_metrics_tail = state["metrics"]["tail"]

    for turn in range(1, args.turns + 1):
        before_counts = trio_turn_counts(recent_metrics_tail)
        api_input = build_api_input(
            turn_index=turn,
            turns=args.turns,
            prior_prompts=prior_prompts,
            recent_log=recent_log_context,
            metrics_tail=recent_metrics_tail,
        )
        response = openai_response(
            api_key=api_key,
            model=args.model,
            instructions=PROBE_INSTRUCTIONS,
            input_text=api_input,
            max_output_tokens=args.max_output_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
        )
        raw_prompt = extract_output_text(response)
        prompt = sanitize_prompt(raw_prompt, max_chars=args.max_prompt_chars, fallback_index=turn - 1)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        for key in usage_totals:
            val = usage.get(key)
            if isinstance(val, int):
                usage_totals[key] += val

        print(f"[{turn}/{args.turns}] {prompt}", flush=True)
        append_jsonl(events_path, {
            "event": "gpt_turn",
            "iso": now_iso(),
            "turn": turn,
            "response_id": response.get("id"),
            "model": response.get("model"),
            "usage": usage,
            "raw_prompt": raw_prompt,
            "prompt": prompt,
        })
        append_text(transcript_path, f"## Turn {turn}\n\n### GPT user turn\n\n{prompt}\n\n")

        send_to_screen(
            host=args.host,
            screen=args.screen,
            screen_window=args.screen_window,
            prompt=prompt,
            timeout=args.ssh_timeout,
        )
        append_jsonl(
            events_path,
            {
                "event": "sent_to_screen",
                "iso": now_iso(),
                "turn": turn,
                "screen": args.screen,
                "screen_window": args.screen_window,
                "prompt": prompt,
            },
        )
        prior_prompts.append(prompt)

        wait_info: dict[str, Any] = {
            "mode": "fixed_sleep" if args.no_wait_for_trio else "wait_for_trio",
            "before_counts": before_counts,
            "completed": False,
            "timed_out": False,
            "elapsed_seconds": 0.0,
        }
        if args.no_wait_for_trio:
            sleep_with_progress(args.settle_seconds)
        else:
            wait_started = time.monotonic()
            turn_timeout_seconds = max(args.turn_timeout_seconds, 1)
            deadline = wait_started + turn_timeout_seconds
            poll_seconds = max(args.poll_seconds, 1)
            while True:
                elapsed = max(time.monotonic() - wait_started, 0.0)
                state = remote_state(
                    host=args.host,
                    live_dir=args.live_dir,
                    log_offset=log_offset,
                    metrics_line_offset=metrics_line_offset,
                    max_delta_bytes=args.max_delta_bytes,
                    metrics_tail_lines=args.metrics_tail_lines,
                    timeout=args.ssh_timeout,
                )
                if state["metrics"]["abs_path"] != current_metrics_file:
                    current_metrics_file = state["metrics"]["abs_path"]
                    state = remote_state(
                        host=args.host,
                        live_dir=args.live_dir,
                        log_offset=log_offset,
                        metrics_line_offset=0,
                        max_delta_bytes=args.max_delta_bytes,
                        metrics_tail_lines=args.metrics_tail_lines,
                        timeout=args.ssh_timeout,
                    )
                after_counts = trio_turn_counts(state["metrics"]["tail"])
                wait_info.update({
                    "after_counts": after_counts,
                    "elapsed_seconds": round(elapsed, 3),
                })
                if trio_turn_advanced(before_counts, after_counts):
                    wait_info["completed"] = True
                    if args.post_completion_settle_seconds > 0:
                        sleep_with_progress(args.post_completion_settle_seconds)
                        state = remote_state(
                            host=args.host,
                            live_dir=args.live_dir,
                            log_offset=log_offset,
                            metrics_line_offset=metrics_line_offset,
                            max_delta_bytes=args.max_delta_bytes,
                            metrics_tail_lines=args.metrics_tail_lines,
                            timeout=args.ssh_timeout,
                        )
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wait_info["timed_out"] = True
                    break
                time.sleep(min(poll_seconds, remaining))
        turn_waits.append(wait_info)

        # If the metrics file rotated, reset the line offset for that file.
        state = remote_state(
            host=args.host,
            live_dir=args.live_dir,
            log_offset=log_offset,
            metrics_line_offset=metrics_line_offset,
            max_delta_bytes=args.max_delta_bytes,
            metrics_tail_lines=args.metrics_tail_lines,
            timeout=args.ssh_timeout,
        )
        if state["metrics"]["abs_path"] != current_metrics_file:
            current_metrics_file = state["metrics"]["abs_path"]
            state = remote_state(
                host=args.host,
                live_dir=args.live_dir,
                log_offset=log_offset,
                metrics_line_offset=0,
                max_delta_bytes=args.max_delta_bytes,
                metrics_tail_lines=args.metrics_tail_lines,
                timeout=args.ssh_timeout,
            )

        log_delta = state["live_log"]["delta"]
        metric_delta_lines = state["metrics"]["delta"]
        all_metric_lines.extend(metric_delta_lines)
        turn_prefix = f"turn-{turn:02d}"
        write_text(out_dir / f"{turn_prefix}.live.log", log_delta)
        write_text(out_dir / f"{turn_prefix}.metrics.jsonl", "\n".join(metric_delta_lines) + ("\n" if metric_delta_lines else ""))
        append_text(raw_log_path, f"\n===== turn {turn}: {prompt} =====\n{log_delta}\n")
        if metric_delta_lines:
            append_text(metrics_path, "\n".join(metric_delta_lines) + "\n")
        append_text(process_path, f"\n===== turn {turn} {now_iso()} =====\n" + "\n".join(state.get("processes", [])) + "\n")

        metrics_summary = summarise_metrics(metric_delta_lines)
        append_text(transcript_path, "### Live log delta\n\n```text\n" + log_delta + "\n```\n\n")
        append_text(transcript_path, "### Turn wait\n\n```json\n" + json.dumps(wait_info, ensure_ascii=False, indent=2) + "\n```\n\n")
        append_text(transcript_path, "### Metrics summary\n\n```json\n" + json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n```\n\n")
        append_jsonl(events_path, {
            "event": "observed",
            "iso": now_iso(),
            "turn": turn,
            "live_log": {k: v for k, v in state["live_log"].items() if k != "delta"},
            "metrics": {k: v for k, v in state["metrics"].items() if k not in {"delta", "tail"}},
            "metrics_summary": metrics_summary,
            "wait": wait_info,
            "processes": state.get("processes", []),
        })

        log_offset = int(state["live_log"]["size"])
        metrics_line_offset = int(state["metrics"]["line_count"])
        recent_log_context = log_delta or recent_log_context
        recent_metrics_tail = state["metrics"]["tail"]

    final_summary = {
        "finished": now_iso(),
        "host": args.host,
        "live_dir": args.live_dir,
        "screen": args.screen,
        "screen_window": args.screen_window,
        "model": args.model,
        "turns": args.turns,
        "settle_seconds": args.settle_seconds,
        "wait_for_trio": not args.no_wait_for_trio,
        "turn_timeout_seconds": args.turn_timeout_seconds,
        "turn_waits": turn_waits,
        "turns_completed_before_next_prompt": sum(1 for item in turn_waits if item.get("completed")),
        "turn_timeouts": sum(1 for item in turn_waits if item.get("timed_out")),
        "usage_totals": usage_totals,
        "metrics_summary": summarise_metrics(all_metric_lines),
        "out_dir": str(out_dir),
    }
    write_text(out_dir / "summary.json", json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n")
    append_text(transcript_path, "## Final summary\n\n```json\n" + json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n```\n")
    append_jsonl(events_path, {"event": "finish", **final_summary})
    print(f"[done] wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
