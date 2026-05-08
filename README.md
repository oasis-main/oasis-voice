# oasis-voice

A self-hosted, license-clean speech sidecar for openclaw / oasis-claw agents
(and the per-tenant backend for oasis-ai's hosted speech endpoints).

Single tier-resolver, single HTTP/WS surface, swappable STT and TTS backends.
The same server runs as a 200 MB CPU-only container for the cloud MVP and
as a 16 GB GPU container for high-quality real-time work — only the
`VOICE_STT_TIER` / `VOICE_TTS_TIER` env vars change.

## Status

**Sketch.** Directory structure and interfaces are in place; backend adapters
are stubbed. Not runnable yet. Tracking the design conversation in
[oasis-claw](https://github.com/oasis-x/oasis-claw) (the consumer side).

## Why a separate repo

oasis-claw is a Node/TypeScript plugin layer. Speech-model inference is
Python-first (PyTorch, vLLM, ONNX, sherpa-onnx). Pinning a Python sub-tree
inside oasis-claw would couple two release cycles that don't share concerns.

This repo is intentionally only the inference server. It does not know about
phone calls, telephony providers, or agents — voice-call's `providers/base.ts`
in openclaw is the thing that talks to it over HTTP/WS.

## License posture

**Strict permissive only.** Apache 2.0, MIT, or CC-BY-4.0 weights/code. No
CC-BY-NC, no CPML, no model with a footnote. If you find a non-commercial
dependency in here, that is a bug, not a feature flag.

| Tier | STT model | License | TTS model | License |
|---|---|---|---|---|
| `lite` | Moonshine Base | MIT | Piper | MIT |
| `default` | Distil-Whisper / Whisper-small | MIT | Kokoro-82M | Apache-2.0 |
| `stream-light` | Sherpa-ONNX Streaming Zipformer | Apache-2.0 | — | — |
| `natural` | — | — | MeloTTS | MIT |
| `style` | — | — | StyleTTS 2 (gruut variant) | MIT |
| `clone-light` | — | — | OpenVoice v2 | MIT |
| `stream-pro` | Voxtral-Mini-4B-Realtime | Apache-2.0 | — | — |
| `accuracy` | NVIDIA Parakeet TDT 0.6B v2 | CC-BY-4.0 | — | — |
| `clone-pro` | — | — | CosyVoice 2 | Apache-2.0 |

Tiers are independent for STT and TTS — pick `VOICE_STT_TIER=lite` and
`VOICE_TTS_TIER=clone-pro` if that's what your hardware budget says.

## HTTP / WebSocket surface

```
GET  /healthz                          → liveness + active tiers
GET  /v1/tiers                         → list of available tiers, what's loaded
POST /v1/stt/transcribe   (multipart)  → audio file → text  (sync)
WS   /v1/stt/stream                    → audio frames → partial+final transcripts
POST /v1/tts/speak        (json)       → text → audio file (sync)
WS   /v1/tts/stream                    → text → audio frames (TTFA-optimised)
POST /v1/voice/clone      (multipart)  → 10s reference → voice_id
GET  /v1/voices                        → list preset + cloned voices
```

The provider on the openclaw side (`extensions/voice-call/src/providers/`)
treats this as one more telephony-shaped backend — same call lifecycle as
Twilio, but the "media stream" goes to localhost or to oasis-ai instead of
to a carrier.

## Running

```bash
# CPU-only MVP (Piper + Moonshine)
docker compose --profile cpu up

# GPU sidecar for Nimbus (Voxtral-Realtime + CosyVoice 2)
VOICE_STT_TIER=stream-pro VOICE_TTS_TIER=clone-pro \
  docker compose --profile gpu up
```

## Repo layout

```
oasis-voice/
├── README.md                            ← this file
├── pyproject.toml                       ← deps split by tier (extras)
├── docker/
│   ├── Dockerfile.cpu                   ← lite + default tiers, ~200 MB
│   ├── Dockerfile.gpu                   ← stream-pro + clone-pro, CUDA + vLLM
│   └── docker-compose.yml
├── src/oasis_voice/
│   ├── main.py                          ← FastAPI entrypoint
│   ├── config.py                        ← env → ActiveTiers
│   ├── tiers.py                         ← declarative tier registry
│   ├── audio.py                         ← resample, framing, format conversion
│   ├── voices.py                        ← preset + cloned voice registry
│   ├── streaming.py                     ← WS framing, backpressure
│   ├── stt/
│   │   ├── base.py                      ← STTBackend ABC
│   │   ├── moonshine.py                 ← lite
│   │   ├── whisper.py                   ← default
│   │   ├── sherpa_zipformer.py          ← stream-light
│   │   ├── voxtral_realtime.py          ← stream-pro (GPU)
│   │   └── parakeet.py                  ← accuracy
│   └── tts/
│       ├── base.py                      ← TTSBackend ABC
│       ├── piper.py                     ← lite
│       ├── kokoro.py                    ← default
│       ├── melotts.py                   ← natural
│       ├── styletts2.py                 ← style
│       ├── openvoice_v2.py              ← clone-light
│       └── cosyvoice2.py                ← clone-pro (GPU)
├── tests/
└── scripts/
    ├── download-weights.py              ← per-tier weight fetch with sha256 pinning
    └── smoke-call.py                    ← health + sample STT/TTS round-trip
```

## Open design questions

1. **Weight pinning.** The `sherpa-onnx-tts` audit surfaced unverified-checksum
   downloads as a real risk. Every tier in `tiers.py` declares a `revision`
   and `sha256`; `scripts/download-weights.py` refuses to write any weight
   file whose sha256 doesn't match. Same defense, applied consistently.

2. **vLLM coupling.** `stream-pro` (Voxtral-Realtime) requires vLLM with a
   custom architecture; `clone-pro` requires `vllm-omni`. These can't share
   a single Python process cleanly. Likely answer: GPU profile runs them as
   separate subprocesses, oasis-voice proxies to each via local HTTP.

3. **Voice cloning consent.** Cloned voices need a recorded consent step —
   we'll not allow `/v1/voice/clone` to succeed without an attestation token.
   That's an oasis-ai-layer concern, not a model-layer concern, but worth
   flagging here so the API shape allows it.
