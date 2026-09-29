# SPDX-License-Identifier: MIT (this file only; model weights carry their own licences)
"""Per-stage benchmark for the AVTR-1 pipeline.

Wraps every engine with a CUDA-synchronised timer, so the numbers are real GPU time
rather than enqueue time (TensorRT is asynchronous — without the sync you get ~9 ms
for a 350 ms render).

Reports where the time goes, and what a multi-GPU split would buy. A chunk is 5
frames = 200 ms of video, so real-time factor = 200 / ms_per_chunk.

    python benchmark.py [--chunks 25] [--avatar maria] [--size 540x960]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
from generate_offline import (  # noqa: E402
    _align_tracks, _chunk_step, _chunk_window, _load_mono_16k, _slice_chunks,
)

from avtr1_renderer.pipeline import Pipeline  # noqa: E402
from avtr1_renderer.types import Chunk, RenderOptions  # noqa: E402

MOTION_STAGES = {"hubert", "avtr1 encode", "avtr1 decode"}
T: dict[str, list[float]] = {}


class Timer:
    def __init__(self, inner, name):
        self._i, self.name = inner, name
        T.setdefault(name, [])

    def allocate_outputs(self, shapes=None):
        return self._i.allocate_outputs(shapes)

    def __call__(self, x, out=None):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = self._i(x, out=out)
        torch.cuda.synchronize()
        T[self.name].append((time.perf_counter() - t0) * 1e3)
        return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunks", type=int, default=25, help="measured chunks (after warmup)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--avatar", default="maria")
    ap.add_argument("--size", default="540x960", help="HxW render size")
    ap.add_argument("--speech", default="example/speaker_1.ogg")
    args = ap.parse_args()
    h, w = (int(v) for v in args.size.split("x"))

    print("GPU:", torch.cuda.get_device_name(0),
          f"| devices: {torch.cuda.device_count()}")
    p, reg = Pipeline.from_artifacts(avatar_ids=[args.avatar], out_size=(h, w))
    av, mg = reg[args.avatar], p._motion_generator

    mg._hubert = Timer(mg._hubert, "hubert")
    mg._encode = Timer(mg._encode, "avtr1 encode")
    mg._decode = Timer(mg._decode, "avtr1 decode")
    p._stitch = Timer(p._stitch, "stitch")
    p._warp = Timer(p._warp, "warp")
    p._decoder = Timer(p._decoder, "decoder")
    p._matting = Timer(p._matting, "modnet")

    sp, ls = _align_tracks(_load_mono_16k(Path(args.speech)), None, 20.0)
    win, step = _chunk_window(p), _chunk_step(p)
    cs, cl = _slice_chunks(sp, win, step), _slice_chunks(ls, win, step)
    opts = RenderOptions(pixel_format="yuv_i420", bg_id="plain_white", stream_frames=True)

    state, chunk_ms = p.initial_state(av), []
    for i in range(args.warmup + args.chunks):
        if i == args.warmup:
            for k in T:
                T[k].clear()
            chunk_ms = []
        t0 = time.perf_counter()
        state, it = p.process_chunk(
            av, Chunk(audio_speech=cs[i % len(cs)], audio_listen=cl[i % len(cl)]), state, opts)
        for _ in it:
            pass
        torch.cuda.synchronize()
        chunk_ms.append((time.perf_counter() - t0) * 1e3)

    mean = statistics.mean(chunk_ms)
    motion = sum(statistics.mean(v) * len(v) / args.chunks
                 for k, v in T.items() if k in MOTION_STAGES)
    render = sum(statistics.mean(v) * len(v) / args.chunks
                 for k, v in T.items() if k not in MOTION_STAGES)

    print(f"\n{'stage':16s} {'calls':>6} {'ms/call':>9} {'ms/chunk':>9} {'share':>7}")
    print("-" * 52)
    for k, v in T.items():
        calls = len(v) / args.chunks
        per = statistics.mean(v)
        print(f"{k:16s} {calls:6.1f} {per:9.2f} {per * calls:9.1f} "
              f"{100 * per * calls / mean:6.1f}%")
    print(f"{'torch/pack':16s} {'':6s} {'':9s} {mean - motion - render:9.1f} "
          f"{100 * (mean - motion - render) / mean:6.1f}%")
    print("-" * 52)

    srt = sorted(chunk_ms)
    print(f"chunk      mean {mean:6.1f} ms | p50 {srt[len(srt)//2]:6.1f} | "
          f"p95 {srt[int(len(srt)*0.95)]:6.1f}")
    print(f"motion     {motion:6.1f} ms   render {render:6.1f} ms")
    print(f"1 GPU      {mean:6.1f} ms  ->  {200/mean:.2f}x real-time")
    print(f"2 GPUs     {render/2 + motion:6.1f} ms  ->  "
          f"{200/(render/2 + motion):.2f}x real-time   (chunks round-robined)")
    print(f"N GPUs     limit is motion ({motion:.0f} ms) -> "
          f"{200/max(motion, 1e-9):.1f}x ceiling")


if __name__ == "__main__":
    main()
