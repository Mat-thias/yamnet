"""
Listens through the microphone and shows whether the light (grid power) is on.

    python powermap/serving/listen.py [--port 8000] [--device cpu] [--model FILE]

Open http://localhost:8000 and press Start. The page records the microphone and sends the
audio here every 0.48s, and each time the newest 0.975s, one YAMNet patch, is scored:

  - the NEPA alarm sounding (in 2 of the last 3 patches) means the light just came ON
  - a generator starting (running in 70% of the last 21 patches, ~10s) means it is OFF

Otherwise the light keeps its last state. A generator that is still running after the alarm,
as people take a while to switch it off, does not turn the light back off.
"""
import os
import sys
import json
import time
import argparse
from collections import deque
from typing import Dict, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import librosa
import numpy as np
import torch

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
POWERMAP_DIR = os.path.dirname(SERVING_DIR)
# detector.py sits one directory up, shared with training
sys.path.append(POWERMAP_DIR)

from detector import load_detector

MODEL_FILE = os.path.join(POWERMAP_DIR, "saved", "detector.pt")   # written by training/train.py
PAGE_FILE = os.path.join(SERVING_DIR, "listen.html")
SAMPLE_RATE = 16000     # what YAMNet takes
PATCH_SAMPLES = 15600   # 0.975s, one YAMNet patch


class LightState:
    """
    Turns each patch's alarm and generator decisions into whether the light is on.

    The alarm sounds once enough of the last few patches fire. A generator starts once most
    of a longer window fires and stops once few do, the gap between the two keeps a patch
    or two of doubt from flipping it back and forth.
    """

    def __init__(self, alarm_window:int = 3, alarm_needed:int = 2, gen_window:int = 21, gen_start:float = 0.7, gen_stop:float = 0.3):
        self.alarm = deque(maxlen=alarm_window)
        self.alarm_needed = alarm_needed
        self.gen = deque(maxlen=gen_window)
        self.gen_start = gen_start
        self.gen_stop = gen_stop

        self.alarm_sounding = False
        self.gen_running = False
        self.light: Optional[bool] = None   # None until something is heard
        self.since: Optional[float] = None
        self.reason = "nothing heard yet"
        self.events = deque(maxlen=20)

    def update(self, alarm:bool, gen:bool):
        self.alarm.append(alarm)
        self.gen.append(gen)

        if not self.alarm_sounding and sum(self.alarm) >= self.alarm_needed:
            self.alarm_sounding = True
            self.set_light(True, "NEPA alarm heard")
        elif self.alarm_sounding and not any(self.alarm):
            self.alarm_sounding = False

        gen_share = sum(self.gen) / self.gen.maxlen
        if not self.gen_running and len(self.gen) == self.gen.maxlen and gen_share >= self.gen_start:
            self.gen_running = True
            self.set_light(False, "generator started")
        elif self.gen_running and gen_share <= self.gen_stop:
            self.gen_running = False
            self.log("generator stopped")

    def set_light(self, light:bool, reason:str):
        self.log(f"{reason}: light {'on' if light else 'off'}")
        if light != self.light:
            self.light, self.since = light, time.time()
        self.reason = reason

    def log(self, text:str):
        self.events.appendleft({"time": time.time(), "text": text})


class Listener:
    """Keeps the newest audio, scores its last patch and updates the light state."""

    def __init__(self, model_file:str, device:str):
        self.detector, saved = load_detector(model_file, device)
        self.classes, self.thresholds = saved["classes"], saved["thresholds"]
        self.device = device
        self.buffer, self.buffer_sr = np.zeros(0, dtype=np.float32), None
        self.state = LightState()

    @torch.no_grad()
    def hear(self, chunk:np.ndarray, sr:int) -> Dict:
        """Takes the next chunk of microphone audio at sr and returns everything the page shows."""
        if sr != self.buffer_sr:
            self.buffer, self.buffer_sr = np.zeros(0, dtype=np.float32), sr
        # a little more than one patch: the margin takes the resampler's edge at the start
        keep = int(1.25 * PATCH_SAMPLES * sr / SAMPLE_RATE)
        self.buffer = np.concatenate([self.buffer, chunk])[-keep:]
        level_db = float(10 * np.log10(np.mean(chunk ** 2) + 1e-10)) if len(chunk) else None

        probs = None
        if len(self.buffer) == keep:
            audio = self.buffer if sr == SAMPLE_RATE else librosa.resample(self.buffer, orig_sr=sr, target_sr=SAMPLE_RATE)
            patch = torch.from_numpy(np.ascontiguousarray(audio[-PATCH_SAMPLES:])).unsqueeze(0).to(self.device)
            probs = dict(zip(self.classes, torch.sigmoid(self.detector(patch)["logits"][0, 0]).cpu().tolist()))
            self.state.update(alarm=probs["alarm"] >= self.thresholds["alarm"], gen=probs["gen"] >= self.thresholds["gen"])

        return {
            "probs": probs,
            "thresholds": self.thresholds,
            "level_db": level_db,
            "light": self.state.light,
            "since": self.state.since,
            "reason": self.state.reason,
            "alarm_sounding": self.state.alarm_sounding,
            "generator_running": self.state.gen_running,
            "events": list(self.state.events),
        }


class Handler(BaseHTTPRequestHandler):
    """GET / is the page, POST /audio?sr=<rate> takes float32 microphone samples."""

    def do_GET(self):
        if urlparse(self.path).path != "/": return self.send_error(404)
        with open(PAGE_FILE, "rb") as f:
            self.send(f.read(), "text/html; charset=utf-8")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/audio": return self.send_error(404)
        sr = int(parse_qs(url.query)["sr"][0])
        chunk = np.frombuffer(self.rfile.read(int(self.headers["Content-Length"])), dtype="<f4")
        self.send(json.dumps(self.server.listener.hear(chunk, sr)).encode(), "application/json")

    def send(self, body:bytes, content_type:str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # a request every 0.48s would flood the terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu", help="one patch at a time is quick on the cpu")
    parser.add_argument("--model", default=MODEL_FILE, help="a detector saved by training/train.py")
    args = parser.parse_args()

    # 127.0.0.1 only: the page and the microphone audio stay on this machine
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.listener = Listener(args.model, args.device)
    print(f"open http://localhost:{args.port} and press Start, ctrl+c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
