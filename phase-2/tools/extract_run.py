#!/usr/bin/env python3
"""Summarise one run dir -> summary-row.csv (and print one CSV line).

Usage: python tools/extract_run.py runs/<run-dir>
"""
import argparse, csv, re, statistics
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def prom(path):
    """Sum every labelled series per metric name."""
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(errors="ignore").splitlines():
        if not line or line[0] == "#":
            continue
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def delta(b, a, n):
    return a.get(n, 0.0) - b.get(n, 0.0)


def hist_mean_ms(b, a, name):
    s, c = delta(b, a, name + "_sum"), delta(b, a, name + "_count")
    return (s / c * 1000.0) if c else None


def parse_trie(text):
    text = ANSI.sub("", text)
    row = {}
    for line in text.splitlines():
        if "benchmark complete" in line:
            kv = dict(re.findall(r"(\w+)=(\S+)", line))
            def f(k):
                try:
                    return float(kv[k])
                except (KeyError, ValueError):
                    return None
            row = {
                "latency_p50_s": f("latency_s_p50"),
                "latency_p99_s": f("latency_s_p99"),
                "completion_tok_s": f("completion_tok_per_s"),
                "steady_completion_tok_s": f("steady_state_completion_tok_per_s"),
                "prompt_tok_s": f("prompt_tok_per_s"),
                "cached_prompt_tok_s": f("cached_prompt_tok_per_s"),
                "cache_hit_pct": f("cache_hit_rate_pct_mean"),
                "eligible_cache_hit_pct": f("eligible_cache_hit_rate_pct_mean"),
                "failed": f("failed_requests"),
                "wall_s": f("wall_time_s"),
            }
            break
    m = re.search(r"completed=(\d+)/(\d+)", text)
    row["traces_done"] = f"{m.group(1)}/{m.group(2)}" if m else None
    m = re.search(r"concurrency=(\d+) duration=([\d.]+)", text)
    row["concurrency"] = m.group(1) if m else None
    row["duration_s"] = m.group(2) if m else None
    return row


def parse_dmon(path):
    pwr = []
    if Path(path).exists():
        for line in open(path):
            p = line.split()
            if len(p) < 3 or p[0].startswith("#"):
                continue
            try:
                pwr.append(float(p[1]))
            except ValueError:
                pass
    return pwr


ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
d = Path(ap.parse_args().run_dir)

b = prom(d / "metrics-before.txt")
a = prom(d / "metrics-after.txt")
q = delta(b, a, "vllm:prefix_cache_queries_total")
h = delta(b, a, "vllm:prefix_cache_hits_total")
acc = delta(b, a, "vllm:spec_decode_num_accepted_tokens_total")
drf = delta(b, a, "vllm:spec_decode_num_draft_tokens_total")
nd = delta(b, a, "vllm:spec_decode_num_drafts_total")
pwr = parse_dmon(d / "dmon.csv")
active_pwr = [p for p in pwr if p > 100]  # ignore idle samples

row = {"run": d.name}
row.update(parse_trie((d / "trie.log").read_text(errors="ignore")) if (d / "trie.log").exists() else {})
row.update({
    "server_ttft_mean_ms": hist_mean_ms(b, a, "vllm:time_to_first_token_seconds"),
    "server_itl_mean_ms": hist_mean_ms(b, a, "vllm:inter_token_latency_seconds"),
    "server_queue_mean_ms": hist_mean_ms(b, a, "vllm:request_queue_time_seconds"),
    "kv_usage_end": a.get("vllm:kv_cache_usage_perc"),
    "preemptions": delta(b, a, "vllm:num_preemptions_total"),
    "vllm_prefix_hit_pct": (100.0 * h / q) if q else None,
    "spec_accept_rate": (acc / drf) if drf else None,
    "spec_mean_accepted_len": (1 + acc / nd) if nd else None,
    "gpu0_power_active_mean_w": round(statistics.mean(active_pwr), 1) if active_pwr else None,
})

with open(d / "summary-row.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row))
    w.writeheader()
    w.writerow(row)
print(",".join("" if v is None else str(v) for v in row.values()))
