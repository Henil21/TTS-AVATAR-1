# AVTR-1 live — multi-GPU streaming avatar with Indic TTS

Real-time streaming talking-head avatar built on [AVTR-1](https://github.com/avaturn-live/avtr-1),
with [Indic-Mio](https://huggingface.co/SPRINGLab/Indic-Mio) for speech. Type text, the
avatar speaks it, frames stream to the browser as they render.

Measured **1.00–1.05× real-time on 2× NVIDIA T4** — a GPU generation *below* AVTR-1's
stated minimum — by splitting chunk rendering across both cards.

```
text ──► Indic-Mio TTS ──► AVTR-1 motion ──► render (GPU 0 ∥ GPU 1) ──► JPEG/WebSocket ──► browser
```

## Why it's built this way

AVTR-1's renderer is **stateless per chunk** — `render_chunk(motions, avatar, bg)` carries
no state between calls. Only motion generation is autoregressive. So chunks round-robin
across GPUs while motion stays on one, and throughput becomes `render / n_gpus`.

The browser receives the whole utterance's audio up front and paints frame
`floor(audio.currentTime * 25)`. Audio is the clock, so A/V sync is exact no matter how
bursty frame delivery is.

Transport is WebSocket, not WebRTC: WebRTC media needs UDP, and a tunnelled cloud
notebook has none. JPEG frames over a WebSocket traverse any HTTP tunnel.

## Hardware

| GPU | ms/chunk (200 ms of video) | Real-time |
|---|---|---|
| T4 ×1 | 390 | 0.51× |
| **T4 ×2** | **~190** | **1.00–1.05×** |
| RTX 3060 Ti ×1 (upstream) | 207 | 0.97× |
| L40S ×1 (upstream) | 71 | 2.81× |

CUDA 12, TensorRT 10.x, Python 3.12+. Ampere+ is recommended upstream but not required —
Turing (T4, sm75) works; don't pass `--ampere-plus` when building engines.

## Setup

```bash
git clone https://github.com/avaturn-live/avtr-1.git
cd avtr-1
cp /path/to/this/repo/*.py .
cp /path/to/this/repo/requirements.txt .

python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .

export HF_TOKEN=hf_...                       # avtr-1 is a gated repo
export AVTR1_LOCAL_STORAGE=./artifacts

python scripts/download_artifacts.py --workers 4    # ~3.4 GB, once
python scripts/build_avtr1_engines.py               # once, per machine
python scripts/build_hubert_engine.py
python scripts/build_renderer_engines.py
```

TensorRT engines are compute-capability specific — they must be built on the machine
that will serve, and rebuilt if you change GPU model.

### Two gotchas that cost real time

**`libstdc++` too old.** The prebuilt `libgrid_sample_3d_plugin.so` needs `GLIBCXX_3.4.32`
(GCC 13). If the renderer engine build dies with that symbol error, preload a newer one:

```bash
export LD_PRELOAD=/usr/local/lib/julia/libstdc++.so.6.0.33   # Kaggle ships one
# or: add-apt-repository ppa:ubuntu-toolchain-r/test && apt install libstdc++6
```

Keep it set at inference too — the plugin is dlopened again when engines load.

**torchaudio version drift.** MioCodec imports torchaudio; a mismatched build fails with
`undefined symbol: torch_library_impl`. Install torchaudio pinned to your exact torch
version *before* MioCodec.

## Run

```bash
python serve.py --tunnel        # prints a public trycloudflare.com URL
python serve.py                 # local only, http://localhost:8000
python serve.py --stop
tail -f /tmp/avtr1_stream.log
```

Configuration is environment variables:

| Variable | Default | |
|---|---|---|
| `AVTR1_LOCAL_STORAGE` | — | artifacts root (required) |
| `AVTR1_AVATARS` | `maria` | comma-separated, from `reference_frames/` |
| `AVTR1_GPUS` | `2` | render workers to spawn |
| `AVTR1_TTS_DEVICE` | `cuda:0` | where Indic-Mio loads |
| `AVTR1_OUT_SIZE` | `540x960` | render size, `HxW` |
| `AVTR1_IDLE_CHUNKS` | `25` | idle-loop length (25 ≈ 5 s) |
| `AVTR1_PORT` | `8000` | |

Put TTS on the GPU that *isn't* doing the most rendering. With two GPUs, `cuda:0` also
runs motion generation, but that's only ~30 ms/chunk — far cheaper than sharing with a
render worker.

### API

| Endpoint | |
|---|---|
| `POST /api/say` | `{text, avatar, bg}` → `{sid, audio, frames, timing_ms}` |
| `WS /ws/{sid}` | binary frames: 4-byte big-endian index + JPEG. Send `{"i": n}` to report playback position |
| `GET /idle/{avatar}.bin` | packed idle-loop JPEGs (`[len][jpeg]…`) |
| `GET /stats` | counters, per-GPU render times, connected devices |
| `GET /health` | avatars, fps, canvas size |

## Benchmarking

### Per-stage

```bash
python benchmark.py --chunks 25
```

Measured on one T4 at 540×960:

| Stage | calls/chunk | ms/call | ms/chunk | share |
|---|---|---|---|---|
| decoder | 5 | 35.4 | 177.2 | 45% |
| warp | 1 | 139.7 | 139.6 | 36% |
| modnet | 5 | 4.8 | 24.0 | 6% |
| avtr1 decode (ODE) | 4 | 5.8 | 23.2 | 6% |
| hubert | 1 | 5.3 | 5.3 | 1% |
| avtr1 encode | 1 | 0.8 | 0.8 | <1% |
| stitch | 1 | 0.2 | 0.2 | <1% |
| torch/pack | — | — | 20.1 | 5% |
| **total** | | | **390** | **0.51×** |

Motion is only **29 ms** of that. Splitting motion and rendering across two GPUs is
therefore pointless (0.59×); splitting *chunks* across them is what works (~1.05×).

Batching doesn't help on a T4 — 5×b=1 decoder calls take 186.7 ms versus 182.5 ms for one
b=5 call. The card is already compute-saturated by a single call.

### Concurrency

```bash
python loadtest.py --clients 3
```

Observed with two real browsers on separate machines through a Cloudflare tunnel:

| Scenario | per-session real-time factor |
|---|---|
| 1 session | 1.00–1.05× |
| 2 overlapping sessions | 0.84× and 1.02× |

Two users cost ~15% each rather than 50%, because stages pipeline naturally: one session
is in TTS while the other renders. Clients finished with a 1–3 frame lead, so nobody
stalled.

Capacity: total render throughput is ~5.2 chunks/s ≈ **1.04 s of video per second**. A
continuously-talking session consumes 1.0 s/s, so 2× T4 supports **one continuous session**
or **2–3 conversational ones** (real dialogue is bursty).

### Known bottleneck: TTS

Indic-Mio via HuggingFace `generate()` runs at **RTF ≈ 1.3** on a T4 — 4.2–7.1 s to
synthesise 3.5–5.7 s of speech. That is now the dominant latency; rendering is real-time
behind it. Options, roughly in order of payoff:

1. Serve the LLM with vLLM (the model card's recommended path) instead of `generate()`.
2. Chunked synthesis — Indic-Mio emits 25 speech tokens per second of audio, so the first
   ~1 s can be decoded and sent to the renderer while the rest generates.
3. A lighter TTS if Indic language coverage isn't needed.

## Files

| File | |
|---|---|
| `stream_server.py` | FastAPI server, WebSocket streaming, browser UI, telemetry |
| `avatar_worker.py` | per-GPU render worker (priority queue, idle loop, CUDA-synced timing) |
| `tts_engine.py` | Indic-Mio + MioCodec wrapper, zero-shot voice cloning from a reference clip |
| `serve.py` | detached launcher + Cloudflare tunnel |
| `benchmark.py` | per-stage timings and multi-GPU projection |
| `loadtest.py` | N concurrent clients, end-to-end timing |

## Logs

```
06:26:24  server say       sid=84399d avatar=maria chars=46
06:26:30  server tts       sid=84399d ms=5141 audio_s=4.40
06:26:30  gpu0   motion    sid=84399d chunks=22 ms=715 per_chunk=32.5
06:26:30  server queued    sid=84399d chunks=22 motion_ms=754
06:26:31  server join      sid=84399d ip=183.82.127.20 ua=Mozilla/5.0 (Windows NT 10.0
06:26:35  server SUMMARY   sid=84399d chunks=22 wall=4.4s rate=5.00ch/s rt=1.00x jpeg=26ms gpu0=359ms/11 gpu1=373ms/11
06:26:48  server leave     sid=84399d ip=183.82.127.20 frames=110 held=17s
```

One line per event. `SUMMARY` closes each utterance with the honest real-time factor and
per-GPU render times. `STALL` appears only when a client's buffer drops below 10 frames.

## Notes

The idle loop is genuine model output, not a still: silence through the motion generator
produces breathing and blinks, rendered once at startup and ping-ponged (forward then
reverse) so it loops seamlessly.

Running in a notebook, always launch detached (`serve.py` does this via
`start_new_session=True`). Otherwise the server shares the kernel's process group and
interrupting *any* cell kills it.

## Licence

These scripts are MIT. The models are not:

- **AVTR-1** — AVTR-1 Community License, non-commercial by default; commercial use for
  entities under $10M revenue pending replacement of InsightFace.
- **InsightFace** (SCRFD detector, 2D106 landmarks) — non-commercial research only.
- **Indic-Mio** — Apache-2.0. **MioCodec** — MIT.

Read `LICENSE-MODEL.md` and `THIRD-PARTY-NOTICES.md` in the upstream repo before shipping.
