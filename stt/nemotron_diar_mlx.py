"""Nemotron-3-Diarization in MLX, ported from the 🤗 Transformers implementation.

Single file; depends on `mlx` and `numpy` (plus `huggingface_hub` only to download). Loads NVIDIA's
Transformers-format checkpoint (`config.json` + `model.safetensors` of `nvidia/Nemotron-3-Diarization`)
directly. The weights are bf16-origin, so `dtype=mx.bfloat16` is lossless.

Ported from transformers `models/nemotron3_diarization/modeling_nemotron3_diarization.py`
(Nemotron3DiarizationForAudioFrameClassification, Nemotron3DiarizationSpeakerCache) and
`models/nemotron_asr_streaming/feature_extraction_nemotron_asr_streaming.py` (Apache-2.0). The model
weights are NVIDIA's, under the OpenMDW License Agreement v1.1. Imported only inside the stt runtime
(see `runtime_worker.py`); the CLI environment never imports it.

    model = Nemotron3Diarizer.from_pretrained()
    probs = model.diarize(audio)                        # (len(audio) // 160, 8) at 10 ms, offline preset
    for block_probs in model.stream(pcm_blocks, "low"):  # live, 1.04 s input-buffer latency
        ...
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# (chunk_len, right_context, fifo_len, update_period) in 80 ms encoder frames: NVIDIA's model-card
# presets, then FluidAudio's FIFO-40 presets (Nemotron3Types.swift, v0.17.1).
PRESETS = {
    "offline": (340, 40, 40, 300),
    "low": (9, 4, 264, 222),
    "very_low": (6, 2, 264, 222),
    "ultra_low": (3, 1, 264, 222),
    "fast": (9, 4, 40, 40),
    "fast32": (32, 4, 40, 40),
    "fast128": (128, 4, 40, 40),
}
SAMPLE_RATE, HOP, N_FFT, WIN, N_MELS, PREEMPH, LOG_GUARD = 16000, 160, 512, 400, 128, 0.97, 2.0**-24


def slaney_mel_filters(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS) -> np.ndarray:
    """`librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=sr/2, norm="slaney")`."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0

    def hz_to_mel(f):
        return min_log_mel + np.log(f / min_log_hz) / logstep if f >= min_log_hz else f / f_sp

    def mel_to_hz(m):
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_freqs[None, :]
    weights = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    return weights.astype(np.float32)


class LogMel:
    """Preemphasis 0.97, symmetric Hann(400) centered in a 512-point FFT, power, slaney mel, log(x + 2^-24).

    `frames(audio, start, count, total, offset)` returns frames [start, start + count) of one
    centered STFT over a signal of `total` samples (`None` while a live signal has no end yet),
    reading `audio` = signal[offset:]. Frames at or past `total // HOP` are zero, as the
    Transformers extractor masks them."""

    def __init__(self):
        window = np.pad(np.hanning(WIN), ((N_FFT - WIN) // 2, N_FFT - WIN - (N_FFT - WIN) // 2))
        self.window = mx.array(window.astype(np.float32))
        self.fb = mx.array(slaney_mel_filters().T)  # (257, 128)

    def frames(self, audio: mx.array, start: int, count: int, total: int | None, offset: int = 0) -> mx.array:
        lo = start * HOP - N_FFT // 2 - 1  # one extra sample feeds the preemphasis of the first
        hi = (start + count - 1) * HOP + N_FFT // 2
        end = offset + audio.shape[0] if total is None else min(total, offset + audio.shape[0])
        a0, a1 = max(lo, 0), min(hi, end)
        assert a0 >= offset, "streaming buffer dropped samples that are still needed"
        src = audio[a0 - offset : a1 - offset] if a1 > a0 else mx.zeros((0,), mx.float32)
        seg = mx.pad(src, [(a0 - lo, hi - lo - (a0 - lo) - src.shape[0])])
        y = seg[1:] - PREEMPH * seg[:-1]  # signal indices lo+1 ... hi-1
        if total is not None and hi > total:
            y = mx.where(mx.arange(y.shape[0]) + lo + 1 < total, y, 0.0)
        x = mx.as_strided(y, (count, N_FFT), (HOP, 1))
        spec = mx.fft.rfft(x * self.window)
        mel = mx.log((mx.square(mx.real(spec)) + mx.square(mx.imag(spec))) @ self.fb + LOG_GUARD)
        if total is None or start + count <= total // HOP:
            return mel
        return mx.where((mx.arange(start, start + count) < total // HOP)[:, None], mel, 0.0)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)  # q_proj | k_proj | v_proj, fused at load
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=10000.0)

    def __call__(self, x, mask=None):
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        o = mx.fast.scaled_dot_product_attention(self.rope(q), self.rope(k), v, scale=self.head_dim**-0.5, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, t, d))


class Layer(nn.Module):
    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.self_attn = Attention(dim, heads)
        self.fc1, self.fc2 = nn.Linear(dim, ffn), nn.Linear(ffn, dim)

    def __call__(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.fc2(nn.gelu(self.fc1(self.layer_norm2(x))))


@dataclass
class Config:
    hidden: int = 512
    heads: int = 8
    ffn: int = 2048
    layers: int = 31
    mels: int = 128
    factor: int = 8
    head_hidden: int = 192
    speakers: int = 8
    cache_len: int = 264
    silence_slots: int = 1
    score_threshold: float = 0.25
    latest_boost: float = 0.05
    min_positive_rate: float = 0.5
    strong_boost_rate: float = 0.75
    weak_boost_rate: float = 1.5

    @classmethod
    def from_hf(cls, cfg: dict) -> "Config":
        a, h, s = cfg["audio_config"], cfg["head_config"], cfg["streaming_config"]
        return cls(hidden=a["hidden_size"], heads=a["num_attention_heads"], ffn=a["intermediate_size"],
                   layers=a["num_hidden_layers"], mels=a["num_mel_bins"], factor=a["subsampling_factor"],
                   head_hidden=h["hidden_size"], speakers=h["num_speakers"], cache_len=s["speaker_cache_length"],
                   silence_slots=s["speaker_cache_silence_frames_per_speaker"],
                   score_threshold=s["prediction_score_threshold"], latest_boost=s["latest_frames_score_boost"],
                   min_positive_rate=s["min_positive_scores_rate"], strong_boost_rate=s["strong_boost_rate"],
                   weak_boost_rate=s["weak_boost_rate"])


class Nemotron3Diarizer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        c = self.config = config
        self.embedder = nn.Linear(c.factor * c.mels, c.hidden, bias=False)
        self.input_layer_norm = nn.LayerNorm(c.hidden)
        self.layers = [Layer(c.hidden, c.heads, c.ffn) for _ in range(c.layers)]
        self.layer_norm = nn.LayerNorm(c.hidden)
        self.proj = nn.Linear(c.hidden, c.head_hidden)
        self.upsampler = nn.Conv1d(c.head_hidden, c.head_hidden * c.factor, kernel_size=3, padding=1)
        self.dense = nn.Linear(c.head_hidden, c.head_hidden)
        self.out_proj = nn.Linear(c.head_hidden, c.speakers)
        self.silence_embeds = mx.zeros((c.hidden,))
        self.logmel = LogMel()
        budget = c.cache_len // c.speakers - c.silence_slots
        self.min_positive = math.floor(budget * c.min_positive_rate)
        self.n_strong = math.floor(budget * c.strong_boost_rate)
        self.n_weak = math.floor(budget * c.weak_boost_rate)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, repo_or_dir="nvidia/Nemotron-3-Diarization", dtype=mx.bfloat16, revision=None):
        path = Path(repo_or_dir)
        if not path.exists():
            from huggingface_hub import snapshot_download

            patterns = ["config.json", "model.safetensors"]
            try:
                path = Path(snapshot_download(repo_or_dir, revision=revision, allow_patterns=patterns,
                                              local_files_only=True))
            except Exception:  # not cached yet
                path = Path(snapshot_download(repo_or_dir, revision=revision, allow_patterns=patterns))
        model = cls(Config.from_hf(json.loads((path / "config.json").read_text())))
        return model.load_hf_weights(mx.load(str(path / "model.safetensors")), dtype)

    def load_hf_weights(self, w: dict, dtype):
        t = "model.audio_tower."
        out = {
            "embedder.weight": w[t + "embedder.projection.weight"],
            "input_layer_norm.weight": w[t + "input_layer_norm.weight"],
            "input_layer_norm.bias": w[t + "input_layer_norm.bias"],
            "layer_norm.weight": w[t + "layer_norm.weight"],
            "layer_norm.bias": w[t + "layer_norm.bias"],
            "proj.weight": w["model.proj.weight"],
            "proj.bias": w["model.proj.bias"],
            "upsampler.weight": w["model.upsampler.conv.weight"].transpose(0, 2, 1),  # (out,in,k) -> (out,k,in)
            "upsampler.bias": w["model.upsampler.conv.bias"],
            "dense.weight": w["classifier.dense.weight"],
            "dense.bias": w["classifier.dense.bias"],
            "out_proj.weight": w["classifier.out_proj.weight"],
            "out_proj.bias": w["classifier.out_proj.bias"],
            "silence_embeds": w["silence_embeds"],
        }
        renames = {"self_attn.o_proj": "self_attn.o_proj", "layer_norm1": "layer_norm1", "layer_norm2": "layer_norm2",
                   "mlp.fc1": "fc1", "mlp.fc2": "fc2"}
        for i in range(self.config.layers):
            src, dst = f"{t}layers.{i}.", f"layers.{i}."
            out[dst + "self_attn.qkv.weight"] = mx.concatenate([w[f"{src}self_attn.{n}_proj.weight"] for n in "qkv"])
            for s, d in renames.items():
                for p in ("weight", "bias"):
                    out[f"{dst}{d}.{p}"] = w[f"{src}{s}.{p}"]
        self.load_weights([(k, v.astype(dtype)) for k, v in out.items()])  # strict: every parameter must be set
        mx.eval(self.parameters())
        return self

    @property
    def dtype(self):
        return self.embedder.weight.dtype

    # ------------------------------------------------------------------ network
    def embed(self, mel: mx.array) -> mx.array:
        """(T, 128) mel -> (ceil(T / 8), 512); the last group is zero-padded like the original."""
        mel = mx.pad(mel, [(0, -mel.shape[0] % self.config.factor), (0, 0)])
        return self.embedder(mel.reshape(-1, self.config.factor * self.config.mels).astype(self.dtype))

    def logits(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """(1, T, 512) packed [cache | fifo | chunk | look-ahead] -> (8T, 8) logits; positions restart per call."""
        h = self.input_layer_norm(x)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.upsampler(self.proj(self.layer_norm(h))).reshape(-1, self.config.head_hidden)
        return self.out_proj(nn.relu(self.dense(nn.relu(h))))

    # ------------------------------------------------------------------ speaker cache: AOSC + FIFO
    def new_state(self, preset: str) -> dict:
        chunk, right, fifo, period = PRESETS[preset]
        c = self.config
        return {"chunk": chunk, "right": right, "fifo_len": fifo, "period": period, "compressed": False,
                "cache": mx.zeros((0, c.hidden), self.dtype), "cache_probs": mx.zeros((0, c.speakers)),
                "fifo": mx.zeros((0, c.hidden), self.dtype)}

    def step(self, state: dict, window: mx.array, n_chunk: int) -> mx.array:
        """Forward one window (`n_chunk` chunk frames then look-ahead frames); returns the chunk's (8 n_chunk, 8) probs."""
        c, f = self.config, self.config.factor
        n_cache, n_fifo = state["cache"].shape[0], state["fifo"].shape[0]
        packed = mx.concatenate([state["cache"], state["fifo"], window])
        probs = mx.sigmoid(self.logits(packed[None]).astype(mx.float32))
        pooled = probs.reshape(-1, f, c.speakers).mean(axis=1)  # encoder-rate probabilities score the cache
        fifo = mx.concatenate([state["fifo"], window[:n_chunk]])
        n = fifo.shape[0]
        if n > state["fifo_len"]:
            pop = min(max(state["period"], n - state["fifo_len"]), n)
            # an uncompressed cache is re-scored every step; a compressed one keeps the probs it was built with
            stored = state["cache_probs"] if state["compressed"] else pooled[:n_cache]
            cache = mx.concatenate([state["cache"], fifo[:pop]])
            cache_probs = mx.concatenate([stored, pooled[n_cache : n_cache + pop]])
            fifo = fifo[pop:]
            if cache.shape[0] > c.cache_len:
                cache, cache_probs = self._compress(cache, cache_probs)
                state["compressed"] = True
            state["cache"], state["cache_probs"] = cache, cache_probs
        state["fifo"] = fifo
        start = (n_cache + n_fifo) * f
        return probs[start : start + n_chunk * f]

    def _boost(self, scores: mx.array, k: int, boost: float) -> mx.array:
        top = mx.argpartition(-scores, kth=k - 1, axis=0)[:k]  # per speaker, its k best frames
        hit = mx.zeros(scores.shape).at[top, mx.arange(scores.shape[1])[None, :]].add(1.0) > 0
        return mx.where(hit, scores + boost, scores)

    def _compress(self, embeds: mx.array, probs: mx.array):
        """Keep `cache_len` frames: grouped by speaker, time-ordered within a speaker, silence slots included."""
        c = self.config
        n = probs.shape[0]
        log_p = mx.log(mx.maximum(probs, c.score_threshold))
        log_q = mx.log(mx.maximum(1.0 - probs, c.score_threshold))
        scores = log_p - log_q + log_q.sum(axis=-1, keepdims=True) - math.log(0.5)
        speech = probs > 0.5
        scores = mx.where(speech, scores, -mx.inf)
        positive = scores > 0
        enough = positive.sum(axis=0, keepdims=True) >= self.min_positive
        scores = mx.where(~positive & speech & enough, -mx.inf, scores)  # drop overlapped frames of well-covered speakers
        scores = mx.where(mx.arange(n)[:, None] >= c.cache_len, scores + c.latest_boost, scores)
        scores = self._boost(scores, self.n_strong, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.n_weak, -math.log(0.5))
        scores = mx.concatenate([scores, mx.full((c.silence_slots, c.speakers), mx.inf)])
        n_scored = n + c.silence_slots
        flat = scores.T.reshape(-1)  # speaker-major, time-minor
        top = mx.argpartition(-flat, kth=c.cache_len - 1)[: c.cache_len]
        sentinel = n_scored * c.speakers
        top = mx.sort(mx.where(flat[top] == -mx.inf, sentinel, top))
        frames = mx.where(top == sentinel, n, mx.minimum(top % n_scored, n))
        embeds = mx.concatenate([embeds, self.silence_embeds.astype(embeds.dtype)[None]])
        probs = mx.concatenate([probs, mx.zeros((1, c.speakers))])
        return embeds[frames], probs[frames]

    # ------------------------------------------------------------------ drivers
    def diarize(self, audio: np.ndarray, preset: str = "offline", block_frames: int = 1 << 15) -> np.ndarray:
        """Whole recording: (L,) float32 16 kHz mono -> (L // 160, 8) speaker probabilities, 10 ms apart."""
        total = int(audio.shape[0])
        n_mel = total // HOP
        if n_mel == 0:
            return np.zeros((0, self.config.speakers), np.float32)
        signal = mx.array(np.ascontiguousarray(audio, dtype=np.float32))
        embeds = []
        for s in range(0, n_mel, block_frames):  # block_frames is a multiple of 8
            embeds.append(self.embed(self.logmel.frames(signal, s, min(block_frames, n_mel - s), total)))
            mx.eval(embeds[-1])
        embeds = mx.concatenate(embeds)
        state, out = self.new_state(preset), []
        chunk, right, n_embeds = state["chunk"], state["right"], embeds.shape[0]
        for s in range(0, n_embeds, chunk):
            e = min(s + chunk, n_embeds)
            out.append(self.step(state, embeds[s : min(e + right, n_embeds)], e - s))
            mx.eval(out[-1], state["cache"], state["cache_probs"], state["fifo"])
        return np.array(mx.concatenate(out)[:n_mel])

    def stream(self, blocks, preset: str = "low"):
        """Live input: iterate 16 kHz float32 PCM blocks of any size; yields (n, 8) probabilities per ready chunk.

        A chunk is ready once its look-ahead and STFT support have arrived. The concatenated output
        equals `diarize(np.concatenate(blocks), preset)` up to float rounding."""
        state = self.new_state(preset)
        f = self.config.factor
        chunk_mel, window_mel = state["chunk"] * f, (state["chunk"] + state["right"]) * f
        buffer, offset, received, done = np.zeros(0, np.float32), 0, 0, 0

        def run(n_frames, count, total):
            nonlocal done
            mel = self.logmel.frames(mx.array(buffer), done, count, total, offset)
            probs = self.step(state, self.embed(mel), -(-n_frames // f))[:n_frames]
            mx.eval(probs, state["cache"], state["cache_probs"], state["fifo"])
            done += n_frames
            return np.array(probs)

        for block in blocks:
            buffer = np.concatenate([buffer, np.asarray(block, np.float32)])
            received += len(block)
            while received >= (done + window_mel - 1) * HOP + N_FFT // 2:
                yield run(chunk_mel, window_mel, None)
                keep = max(0, done * HOP - N_FFT // 2 - 1)
                buffer, offset = buffer[keep - offset :], keep
        while received // HOP > done:  # end of input: the tail is chunked like `diarize`
            left = received // HOP - done
            yield run(min(chunk_mel, left), min(window_mel, left), received)
