# SPDX-License-Identifier: MIT
"""Concurrency test: fire N simultaneous /api/say requests and report per-session timing.

Measures what a real client experiences, including the TTS queue. Run it against a
server that is already up.

    python loadtest.py --clients 3 --url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import websocket  # pip install websocket-client


def one(i: int, url: str, text: str, timeout: int) -> dict:
    t0 = time.perf_counter()
    r = requests.post(f"{url}/api/say", json={"text": text}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    say_s = time.perf_counter() - t0

    ws_url = url.replace("http", "ws", 1) + f"/ws/{j['sid']}"
    ws = websocket.create_connection(ws_url, timeout=timeout)
    got, first_frame_s = 0, None
    try:
        while got < j["frames"]:
            ws.recv()
            got += 1
            if got == 1:
                first_frame_s = time.perf_counter() - t0
    except Exception:
        pass
    finally:
        ws.close()

    total = time.perf_counter() - t0
    video_s = j["frames"] / j["fps"]
    return {"client": i, "frames": got, "expected": j["frames"], "video_s": video_s,
            "tts_ms": j["timing_ms"]["tts"], "motion_ms": j["timing_ms"]["motion"],
            "say_s": say_s, "first_frame_s": first_frame_s, "total_s": total,
            "rt": video_s / total if total else 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clients", type=int, default=3)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--text", default="Hello, this is a concurrency test.")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    print(requests.get(f"{args.url}/health", timeout=10).json())
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.clients) as ex:
        res = list(ex.map(lambda i: one(i, args.url, args.text, args.timeout),
                          range(args.clients)))
    wall = time.perf_counter() - t0

    print(f"\n{'cli':>3} {'frames':>7} {'video_s':>8} {'tts_ms':>7} {'1st_frame':>10} "
          f"{'total_s':>8} {'rt':>6}")
    for r in sorted(res, key=lambda x: x["client"]):
        ff = f"{r['first_frame_s']:.1f}" if r["first_frame_s"] else "-"
        print(f"{r['client']:3d} {r['frames']:3d}/{r['expected']:<3d} {r['video_s']:8.1f} "
              f"{r['tts_ms']:7d} {ff:>10} {r['total_s']:8.1f} {r['rt']:6.2f}")

    tot_video = sum(r["video_s"] for r in res)
    print(f"\nclients={args.clients}  wall={wall:.1f}s  "
          f"video={tot_video:.1f}s  throughput={tot_video / wall:.2f}x real-time")
    print(f"first frame: p50 {statistics.median([r['first_frame_s'] or 0 for r in res]):.1f}s "
          f"max {max((r['first_frame_s'] or 0) for r in res):.1f}s")
    print(json.dumps(requests.get(f"{args.url}/stats", timeout=10).json()["render_ms"]))


if __name__ == "__main__":
    main()
