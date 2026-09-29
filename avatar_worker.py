# SPDX-License-Identifier: MIT (this file only; model weights carry their own licences)
"""Per-GPU render worker.

One process per GPU, pinned with CUDA_VISIBLE_DEVICES. Two facts shape this:

* Motion generation is autoregressive — it carries 75 frames of history plus a noise
  seed — so it must stay in one process (worker 0).
* Rendering is *stateless* per chunk: `render_chunk(motions, avatar, bg)` takes no
  state. So chunks can be round-robined across every GPU you have.

That's what makes two T4s reach real-time: each chunk still costs ~350 ms, but two
render concurrently, so throughput is ~175 ms per 200 ms chunk.

Jobs arrive on two queues. `hi_q` (motion, idle loops) preempts `lo_q` (renders);
without that, a new utterance's motion would queue behind the previous utterance's
renders and the UI would appear to hang.
"""

from __future__ import annotations

import queue
import time

WINDOW = 6480          # (chunk_size + future_size) * frame_len + audio_shift
JPEG_QUALITY = 82


def worker_main(gpu, hi_q, lo_q, out_q, avatars, out_size, ready):
    import os
    import sys

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)      # must precede CUDA init
    sys.path.insert(0, "scripts")

    import cv2
    import numpy as np
    import torch
    from avtr1_renderer.components.liveportrait.motion_stitch import MotionFrame
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.renderer import render_chunk
    from avtr1_renderer.types import Chunk, RenderOptions

    def log(ev, **kv):
        print(f"{time.strftime('%H:%M:%S')}  gpu{gpu}   {ev:<9} "
              + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)

    t0 = time.perf_counter()
    pipe, reg = Pipeline.from_artifacts(avatar_ids=avatars, out_size=out_size)
    opts = RenderOptions(pixel_format="yuv_i420", bg_id="plain_white", stream_frames=False)
    log("ready", load_s=f"{time.perf_counter() - t0:.1f}", size=f"{out_size[1]}x{out_size[0]}")

    def do_render(avatar_id, bg_id, R, exp):
        av = reg[avatar_id]
        m = MotionFrame(R=torch.from_numpy(R).cuda(), exp=torch.from_numpy(exp).cuda())
        rgb, _ = render_chunk(m, av, pipe._backgrounds[bg_id], stitch=pipe._stitch,
                              warp=pipe._warp, decoder=pipe._decoder, matting=pipe._matting)
        torch.cuda.synchronize()          # TensorRT enqueues async; sync to time it honestly
        return rgb

    def to_jpgs(rgb):
        arr = (rgb.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return [cv2.imencode(".jpg", a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1]
                .tobytes() for a in arr]

    def gen_motion(avatar_id, cs, cl):
        av = reg[avatar_id]
        state, motions = pipe.initial_state(av), []
        for s, l in zip(cs, cl):
            m, state = pipe._motion_generator.generate_chunk(
                Chunk(audio_speech=s, audio_listen=l), av, state, opts)
            motions.append((m.R.cpu().numpy(), m.exp.cpu().numpy()))
        return motions

    ready.put(gpu)

    while True:
        try:
            job = hi_q.get_nowait()
        except queue.Empty:
            try:
                job = lo_q.get(timeout=0.05)
            except queue.Empty:
                continue
        if job is None:
            break
        kind = job[0]

        if kind == "idle_loop":
            # Silence through the motion generator gives real idle motion (breathing,
            # blinks). Rendered once at startup; the client ping-pongs it so it loops
            # without a visible seam.
            sid, avatar_id, bg_id, n_chunks = job[1:]
            t = time.perf_counter()
            silence = [np.zeros(WINDOW, dtype=np.float32) for _ in range(n_chunks)]
            jpgs = []
            for R, exp in gen_motion(avatar_id, silence, silence):
                jpgs += to_jpgs(do_render(avatar_id, bg_id, R, exp))
            log("idleloop", avatar=avatar_id, frames=len(jpgs), s=f"{time.perf_counter()-t:.1f}")
            out_q.put(("idle_loop", sid, avatar_id, jpgs))

        elif kind == "motion":
            sid, avatar_id, cs, cl = job[1:]
            t = time.perf_counter()
            motions = gen_motion(avatar_id, cs, cl)
            ms = (time.perf_counter() - t) * 1e3
            log("motion", sid=sid[:6], chunks=len(motions), ms=f"{ms:.0f}",
                per_chunk=f"{ms / max(1, len(motions)):.1f}")
            out_q.put(("motion", sid, motions))

        elif kind == "render":
            sid, idx, R, exp, avatar_id, bg_id = job[1:]
            t = time.perf_counter()
            rgb = do_render(avatar_id, bg_id, R, exp)
            render_ms = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            jpgs = to_jpgs(rgb)
            out_q.put(("frames", sid, idx, jpgs, gpu, render_ms,
                       (time.perf_counter() - t) * 1e3))
