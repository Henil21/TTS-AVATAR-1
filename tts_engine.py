# SPDX-License-Identifier: MIT (this file only; model weights carry their own licences)
"""Indic-Mio TTS wrapper.

SPRINGLab/Indic-Mio is a 0.6B LLM that emits speech tokens (`<|s_NNN|>`); MioCodec
turns those into a waveform. The codec needs a *speaker* embedding, taken here from
a reference clip, so the voice is cloned zero-shot from whatever wav you point at.

Note: the Indic-Mio model card's decode snippet (`codec.decode(codes_tensor)`) does
not match MioCodec's actual API, which is
`decode(global_embedding=..., content_token_indices=...)`.

Returns both the native-rate wav (for browser playback) and a 16 kHz copy, which is
what the avatar pipeline consumes.
"""

from __future__ import annotations

import re

import numpy as np
import soxr
import torch
from miocodec import MioCodecModel, load_audio
from transformers import AutoModelForCausalLM, AutoTokenizer

TOKEN_RE = re.compile(r"<\|s_(\d+)\|>")
LLM_ID = "SPRINGLab/Indic-Mio"
CODEC_ID = "Aratako/MioCodec-25Hz-24kHz"


class IndicMio:
    def __init__(self, device: str = "cuda:0", ref_wav: str = "example/speaker_1.ogg") -> None:
        # Turing (T4, sm75) has no bf16
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(LLM_ID, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=dtype).to(device).eval()
        self.codec = MioCodecModel.from_pretrained(CODEC_ID).eval().to(device)
        self.sr = int(self.codec.config.sample_rate)

        ref = load_audio(ref_wav, sample_rate=self.sr).to(device)
        with torch.no_grad():
            self.spk = self.codec.encode(ref, return_content=False,
                                         return_global=True).global_embedding
        print(f"[tts] ready device={device} codec={self.sr}Hz dtype={dtype} ref={ref_wav}",
              flush=True)

    @torch.no_grad()
    def __call__(self, text: str, max_new_tokens: int = 1024
                 ) -> tuple[np.ndarray, int, np.ndarray]:
        """text -> (wav at self.sr, sample_rate, wav at 16 kHz)."""
        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        inp = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.llm.generate(**inp, max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=0.9, top_p=0.9)
        gen = self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
        codes = [int(x) for x in TOKEN_RE.findall(gen)]
        if not codes:
            raise ValueError("model produced no speech tokens")
        wav = self.codec.decode(
            global_embedding=self.spk,
            content_token_indices=torch.tensor(codes, dtype=torch.long, device=self.device),
        ).squeeze().float().cpu().numpy()
        wav16 = soxr.resample(wav, self.sr, 16000, quality="HQ").astype(np.float32)
        return wav, self.sr, wav16


__all__ = ["IndicMio"]
