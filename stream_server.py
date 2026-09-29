# SPDX-License-Identifier: MIT (this file only; model weights carry their own licences)
"""AVTR-1 live streaming server.

    text -> Indic-Mio TTS -> AVTR-1 motion -> multi-GPU render -> JPEG over WebSocket

The browser gets the whole utterance's audio up front and frames as they render, then
paints frame `floor(audio.currentTime * 25)`. Audio is therefore the clock and A/V
sync is exact regardless of how bursty frame delivery is.

Why not WebRTC: media needs UDP, and a tunnelled Kaggle/Colab box has none. WebSocket
goes through any HTTP tunnel.

Env:
    AVTR1_LOCAL_STORAGE   artifacts root (required by the renderer)
    AVTR1_AVATARS         comma-separated avatar ids           (default: maria)
    AVTR1_TTS_DEVICE      cuda:N for Indic-Mio                 (default: cuda:0)
    AVTR1_OUT_SIZE        HxW render size                      (default: 540x960)
    AVTR1_IDLE_CHUNKS     chunks of idle motion to pre-render   (default: 25 = 5 s)
    AVTR1_PORT            listen port                          (default: 8000)
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import statistics
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import transformers
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from avatar_worker import worker_main

transformers.logging.set_verbosity_error()
from tts_engine import IndicMio  # noqa: E402  (after verbosity is quieted)

OUT = Path(os.environ.get("AVTR1_OUT_DIR", "/tmp/avtr1_stream"))
OUT.mkdir(parents=True, exist_ok=True)
AVATARS = os.environ.get("AVTR1_AVATARS", "maria").split(",")
TTS_DEVICE = os.environ.get("AVTR1_TTS_DEVICE", "cuda:0")
OUT_H, OUT_W = (int(v) for v in os.environ.get("AVTR1_OUT_SIZE", "540x960").split("x"))
IDLE_CHUNKS = int(os.environ.get("AVTR1_IDLE_CHUNKS", "25"))
PORT = int(os.environ.get("AVTR1_PORT", "8000"))
N_GPUS = int(os.environ.get("AVTR1_GPUS", "2"))

WINDOW, STEP, FPS = 6480, 3200, 25      # chunk geometry of the AVTR-1 motion generator


def log(ev, **kv):
    print(f"{datetime.now():%H:%M:%S}  server {ev:<9} "
          + " ".join(f"{k}={v}" for k, v in kv.items()), flush=True)


def slice_chunks(a: np.ndarray) -> list[np.ndarray]:
    n = max(1, (len(a) + STEP - 1) // STEP)
    out = []
    for i in range(n):
        piece = a[i * STEP: i * STEP + WINDOW]
        if len(piece) < WINDOW:
            piece = np.pad(piece, (0, WINDOW - len(piece)))
        out.append(piece.astype(np.float32))
    return out


ctx = mp.get_context("spawn")
HI_Q = [ctx.Queue() for _ in range(N_GPUS)]
LO_Q = [ctx.Queue() for _ in range(N_GPUS)]
OUT_Q, READY = ctx.Queue(), ctx.Queue()

SESS: dict[str, asyncio.Queue] = {}
JOBS: dict[str, dict] = {}
DEVICES: dict[str, dict] = {}
IDLE: dict[str, bytes] = {}
RENDER_MS = defaultdict(lambda: deque(maxlen=200))
JPEG_MS = deque(maxlen=200)
STATS = {"sessions": 0, "frames_sent": 0, "chunks": 0}
LOOP: asyncio.AbstractEventLoop | None = None
TTS: IndicMio | None = None


def pump():
    """mp.Queue (worker results) -> per-session asyncio queues."""
    while True:
        msg = OUT_Q.get()
        if msg is None:
            return
        if msg[0] == "idle_loop":
            IDLE[msg[2]] = b"".join(len(j).to_bytes(4, "big") + j for j in msg[3])
            continue
        q = SESS.get(msg[1])
        if q is not None and LOOP is not None:
            LOOP.call_soon_threadsafe(q.put_nowait, msg)


app = FastAPI()


class Say(BaseModel):
    text: str
    avatar: str = AVATARS[0]
    bg: str = "plain_white"


async def watchdog():
    """Only speaks up when a client is about to run dry."""
    while True:
        await asyncio.sleep(5)
        for sid, d in list(DEVICES.items()):
            j = JOBS.get(sid)
            if not j or j["done"] >= j["total"]:
                continue
            lead = d["frames"] - max(d["last_i"], 0)
            if lead < 10:
                el = time.time() - j["t0"]
                log("STALL", sid=sid[:6], chunks=f"{j['done']}/{j['total']}",
                    rt=f"{j['done'] * 0.2 / max(el, 1e-3):.2f}x",
                    playing=d["last_i"], lead=lead)


@app.on_event("startup")
async def startup():
    global LOOP, TTS
    LOOP = asyncio.get_running_loop()
    threading.Thread(target=pump, daemon=True).start()
    for a in AVATARS:
        HI_Q[0].put(("idle_loop", "-", a, "plain_white", IDLE_CHUNKS))
    log("loading", what="tts", device=TTS_DEVICE)
    TTS = IndicMio(device=TTS_DEVICE)
    asyncio.create_task(watchdog())
    log("ready", port=PORT, avatars=",".join(AVATARS), size=f"{OUT_W}x{OUT_H}")


@app.get("/health")
def health():
    return {"ok": True, "avatars": AVATARS, "fps": FPS, "w": OUT_W, "h": OUT_H}


@app.get("/stats")
def stats():
    return {**STATS, "devices": DEVICES,
            "render_ms": {f"gpu{k}": round(statistics.mean(v), 1)
                          for k, v in RENDER_MS.items() if v},
            "jpeg_ms": round(statistics.mean(JPEG_MS), 1) if JPEG_MS else None}


@app.get("/idle/{avatar}.bin")
def idle(avatar: str):
    blob = IDLE.get(avatar)
    if not blob:
        return JSONResponse({"error": "idle loop still rendering"}, 503)
    return Response(blob, media_type="application/octet-stream")


@app.post("/api/say")
async def say(req: Say):
    sid = uuid.uuid4().hex[:10]
    STATS["sessions"] += 1
    log("say", sid=sid[:6], avatar=req.avatar, chars=len(req.text))

    t0 = time.perf_counter()
    wav, sr, wav16 = await asyncio.to_thread(TTS, req.text)
    sf.write(OUT / f"{sid}.wav", wav, sr)
    tts_ms = (time.perf_counter() - t0) * 1e3
    log("tts", sid=sid[:6], ms=round(tts_ms), audio_s=f"{len(wav) / sr:.2f}")

    cs, cl = slice_chunks(wav16), slice_chunks(np.zeros_like(wav16))
    q: asyncio.Queue = asyncio.Queue()
    SESS[sid] = q

    t1 = time.perf_counter()
    HI_Q[0].put(("motion", sid, req.avatar, cs, cl))     # preempts queued renders
    _, _, motions = await q.get()
    motion_ms = (time.perf_counter() - t1) * 1e3

    for i, (R, exp) in enumerate(motions):               # round-robin across GPUs
        LO_Q[i % N_GPUS].put(("render", sid, i, R, exp, req.avatar, req.bg))
    JOBS[sid] = {"total": len(motions), "done": 0, "t0": time.time(), "ms": defaultdict(list)}
    log("queued", sid=sid[:6], chunks=len(motions), motion_ms=round(motion_ms))

    return {"sid": sid, "audio": f"/files/{sid}.wav", "chunks": len(motions),
            "frames": len(motions) * 5, "fps": FPS,
            "timing_ms": {"tts": round(tts_ms), "motion": round(motion_ms)}}


@app.websocket("/ws/{sid}")
async def ws(sock: WebSocket, sid: str):
    await sock.accept()
    ip = sock.client.host if sock.client else "?"
    ua = sock.headers.get("user-agent", "?")
    DEVICES[sid] = {"ip": ip, "ua": ua[:70], "joined": time.time(), "frames": 0, "last_i": -1}
    log("join", sid=sid[:6], ip=ip, ua=ua.split(")")[0][:40])

    q = SESS.get(sid)
    if q is None:
        await sock.close()
        return

    async def sender():
        while True:
            msg = await asyncio.wait_for(q.get(), timeout=180)
            if msg[0] != "frames":
                continue
            _, _, idx, jpgs, gpu, render_ms, jpeg_ms = msg
            RENDER_MS[gpu].append(render_ms)
            JPEG_MS.append(jpeg_ms)
            STATS["chunks"] += 1
            for k, jpg in enumerate(jpgs):
                await sock.send_bytes((idx * 5 + k).to_bytes(4, "big") + jpg)
            STATS["frames_sent"] += len(jpgs)
            DEVICES[sid]["frames"] += len(jpgs)

            j = JOBS.get(sid)
            if j:
                j["done"] += 1
                j["ms"][gpu].append(render_ms)
                if j["done"] == j["total"]:
                    el = time.time() - j["t0"]
                    per_gpu = {f"gpu{k}": f"{statistics.mean(v):.0f}ms/{len(v)}"
                               for k, v in sorted(j["ms"].items())}
                    log("SUMMARY", sid=sid[:6], chunks=j["total"], wall=f"{el:.1f}s",
                        rate=f"{j['total'] / el:.2f}ch/s", rt=f"{j['total'] * 0.2 / el:.2f}x",
                        jpeg=f"{statistics.mean(JPEG_MS):.0f}ms", **per_gpu)

    async def receiver():
        try:
            while True:
                DEVICES[sid]["last_i"] = json.loads(await sock.receive_text()).get("i", -1)
        except Exception:
            return

    try:
        tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
    except (asyncio.TimeoutError, WebSocketDisconnect):
        pass
    finally:
        d = DEVICES.pop(sid, {})
        log("leave", sid=sid[:6], ip=ip, frames=d.get("frames", 0),
            held=f"{time.time() - d.get('joined', time.time()):.0f}s")
        SESS.pop(sid, None)
        JOBS.pop(sid, None)


@app.get("/files/{name}")
def files(name: str):
    p = OUT / name
    return FileResponse(p) if p.exists() else JSONResponse({"error": "not found"}, 404)


PAGE = """<!doctype html><meta charset=utf-8><title>AVTR-1 live</title>
<style>body{background:#0d0d0d;color:#eee;font:15px system-ui;margin:0;padding:24px;
display:flex;gap:28px;flex-wrap:wrap}.col{max-width:560px}
textarea{width:100%;height:80px;background:#1b1b1b;color:#eee;border:1px solid #2c2c2c;
border-radius:10px;padding:12px;font:15px system-ui;box-sizing:border-box}
button{background:#2d6cdf;color:#fff;border:0;border-radius:10px;padding:11px 20px;
font:15px system-ui;cursor:pointer;margin-top:12px}button:disabled{opacity:.45}
canvas{width:100%;border-radius:12px;background:#000;display:block}
#s{font:12.5px ui-monospace;color:#8fa6bd;white-space:pre-wrap;margin-top:10px}</style>
<div class=col>
  <h2 style=margin-top:0>AVTR-1 live stream</h2>
  <textarea id=x>नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ? &lt;happy&gt;</textarea>
  <button id=go onclick=run()>Speak</button>
  <div id=s>loading…</div>
</div>
<div class=col style=flex:1><canvas id=c></canvas></div>
<script>
const ctx=c.getContext('2d');
let frames=[],au=null,total=0,got=0,ws=null,tick=null;
let idle=[],idleI=0,idleDir=1,idleTimer=null,speaking=false;

(async()=>{
  const h=await (await fetch('/health')).json();
  c.width=h.w; c.height=h.h; s.textContent='rendering idle loop…';
  for(let a=0;a<40;a++){
    const r=await fetch('/idle/'+h.avatars[0]+'.bin');
    if(r.ok){
      const buf=await r.arrayBuffer(), dv=new DataView(buf); let off=0;
      while(off<buf.byteLength){
        const n=dv.getUint32(off); off+=4;
        idle.push(await createImageBitmap(new Blob([buf.slice(off,off+n)],{type:'image/jpeg'})));
        off+=n;
      }
      break;
    }
    await new Promise(r2=>setTimeout(r2,3000));
  }
  s.textContent=idle.length?`ready · idle loop ${idle.length} frames`:'ready (no idle loop)';
  startIdle();
})();

function startIdle(){
  if(idleTimer||!idle.length)return;
  idleTimer=setInterval(()=>{            // ping-pong: forward then reverse, never seams
    if(speaking)return;
    ctx.drawImage(idle[idleI],0,0,c.width,c.height);
    idleI+=idleDir;
    if(idleI>=idle.length-1||idleI<=0)idleDir*=-1;
  },40);
}

function reset(){ if(ws){try{ws.close()}catch(e){}} if(tick)clearInterval(tick);
  frames=[]; got=0; if(au)au.pause(); au=null; }

async function run(){
  go.disabled=true; reset(); s.textContent='synthesising…';
  const r=await fetch('/api/say',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:x.value})});
  const j=await r.json();
  if(j.error){s.textContent='error: '+j.error; go.disabled=false; return}
  total=j.frames;
  s.textContent=`tts ${j.timing_ms.tts}ms · motion ${j.timing_ms.motion}ms · ${total} frames — buffering…`;
  ws=new WebSocket((location.protocol=='https:'?'wss://':'ws://')+location.host+'/ws/'+j.sid);
  ws.binaryType='arraybuffer';
  ws.onmessage=async e=>{
    const i=new DataView(e.data).getUint32(0);
    frames[i]=await createImageBitmap(new Blob([e.data.slice(4)],{type:'image/jpeg'}));
    got++;
    if(got===12&&!au){                    // ~0.5 s buffered, then audio drives the clock
      speaking=true; au=new Audio(j.audio); au.play(); paint();
      tick=setInterval(()=>{ if(ws&&ws.readyState===1&&au)
        ws.send(JSON.stringify({i:Math.floor(au.currentTime*25)})) },1000);
    }
  };
}

function paint(){
  const i=Math.floor(au.currentTime*25);
  if(frames[i]) ctx.drawImage(frames[i],0,0,c.width,c.height);
  s.textContent=`frame ${i}/${total} · buffered ${got} · lead ${got-i}`;
  if(!au.ended) requestAnimationFrame(paint);
  else { s.textContent=`done · ${total} frames`; go.disabled=false; speaking=false;
         if(tick)clearInterval(tick); if(ws)ws.close(); }
}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    procs = [ctx.Process(target=worker_main,
                         args=(g, HI_Q[g], LO_Q[g], OUT_Q, AVATARS, (OUT_H, OUT_W), READY),
                         daemon=True) for g in range(N_GPUS)]
    for p in procs:
        p.start()
    for _ in procs:
        log("worker", gpu=READY.get())
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
