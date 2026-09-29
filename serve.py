# SPDX-License-Identifier: MIT
"""Launch the streaming server detached, optionally behind a Cloudflare tunnel.

`start_new_session=True` matters in notebooks: without it the server is in the
kernel's process group and dies with SIGINT the moment you interrupt any cell.

    python serve.py --tunnel          # public URL via cloudflared
    python serve.py                   # local only
    python serve.py --stop
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SRV_LOG = Path(os.environ.get("AVTR1_LOG", "/tmp/avtr1_stream.log"))
CF_LOG = Path("/tmp/avtr1_cf.log")
CF_BIN = "/usr/local/bin/cloudflared"
CF_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def stop() -> None:
    subprocess.run(["pkill", "-f", "stream_server.py"])
    subprocess.run(["pkill", "-f", "cloudflared tunnel"])
    print("stopped")


def start_server(port: int, timeout: int) -> bool:
    subprocess.run(["pkill", "-f", "stream_server.py"])
    time.sleep(2)
    SRV_LOG.write_bytes(b"")
    subprocess.Popen([sys.executable, str(HERE / "stream_server.py")],
                     cwd=os.environ.get("AVTR1_REPO", "."),
                     stdout=open(SRV_LOG, "wb"), stderr=subprocess.STDOUT,
                     env=os.environ.copy(), start_new_session=True)
    for i in range(timeout):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).ok:
                print(f"server up after {i}s")
                return True
        except Exception:
            pass
        if i % 15 == 0:
            last = (SRV_LOG.read_text(errors="ignore").strip().splitlines() or [""])[-1]
            print(f"{i:4d}s  {last[:110]}", flush=True)
        time.sleep(1)
    print(SRV_LOG.read_text(errors="ignore")[-4000:])
    return False


def start_tunnel(port: int) -> str | None:
    if subprocess.run(["pgrep", "-f", "cloudflared tunnel"], capture_output=True).returncode != 0:
        if not Path(CF_BIN).exists():
            subprocess.run(["wget", "-q", CF_URL, "-O", CF_BIN], check=True)
            os.chmod(CF_BIN, 0o755)
        CF_LOG.write_bytes(b"")
        subprocess.Popen([CF_BIN, "tunnel", "--url", f"http://localhost:{port}",
                          "--no-autoupdate"],
                         stdout=open(CF_LOG, "wb"), stderr=subprocess.STDOUT,
                         start_new_session=True)
    for _ in range(60):
        m = re.findall(r"https://[-\w]+\.trycloudflare\.com",
                       CF_LOG.read_text(errors="ignore"))
        if m:
            return m[-1]
        time.sleep(1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tunnel", action="store_true", help="expose via cloudflared")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.environ.get("AVTR1_PORT", "8000")))
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    if args.stop:
        stop()
        return
    if not start_server(args.port, args.timeout):
        sys.exit(1)
    if args.tunnel:
        url = start_tunnel(args.port)
        print("\n  OPEN:", url or f"tunnel failed — see {CF_LOG}", "\n")
    else:
        print(f"\n  OPEN: http://localhost:{args.port}\n")
    print(f"logs: tail -f {SRV_LOG}")


if __name__ == "__main__":
    main()
