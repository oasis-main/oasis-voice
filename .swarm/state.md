# State — oasis-voice

**Last touched**: 2026-05-11 by Claude (sibling-container shape for
oasis-claw CLAW-021; bootstrap .swarm/ for the repo)
**Current focus**: Lite tier (Piper TTS + Moonshine STT) wired into
oasis-claw's runtime image as a sibling container. Inbound voice
messages from Telegram flow through here as of oasis-claw@d800292.
**Active items**: VOICE-001 (slim the 6.37GB cpu image).
**Branch model**: matches oasis-x convention — push freely to `dev`,
`main` is protected (no CI gates yet, but the discipline matters even
on a brand-new repo).
**Blockers**: None on our side. CLAW-021 outbound voice (oasis-claw
calling our `/v1/tts/speak` and posting the WAV to Telegram via
sendVoice) needs no changes here — the SpeechProvider in the
oasis-claw plugin already speaks this endpoint.

---

## Handoff Note

**2026-05-11 (Claude — ffmpeg-on-server for any-format STT input):**

The /v1/stt/transcribe endpoint previously only accepted WAV (via
soundfile/libsndfile). Telegram voice messages arrive as .ogg/opus,
iMessage clips as .caf/.m4a, browser MediaRecorder output is .webm —
none of which libsndfile decodes by default. Forcing client-side
transcoding would push ffmpeg into every channel adapter (the
oasis-claw plugin first, then future cloud connectors), which is the
wrong shape — the server is where audio-format knowledge belongs.

Commit `436a64f` adds `decode_audio_any()` to `src/oasis_voice/audio.py`
which routes WAV/FLAC through libsndfile (fast path, no subprocess)
and everything else through an ffmpeg subprocess (normalises to PCM_16LE
mono at 16kHz). 30s timeout cap defends against zip-bomb-shaped input.
The endpoint now hands `audio.content_type` to the new helper.

Tests: 7 new cases in `tests/test_audio.py`, including one real-
ffmpeg roundtrip that synthesises an opus blob via ffmpeg and asserts
the decoder produces PCM16. That test is `@needs_ffmpeg`-gated;
it runs inside the container (ffmpeg is in the apt layer per
Dockerfile.cpu:14) and skips on bare hosts. Total: 37 passed, 1
skipped.

oasis-claw's `MediaUnderstandingProvider` (registered in
`extensions/oasis-voice/media-understanding-provider.ts` over there)
POSTs raw .ogg bytes to this endpoint directly — no client-side
transcoding needed.

**Image-size flag**: building `oasis-voice:cpu` locally produced a
6.37GB image. Root cause is `pyproject.toml` line 33's `[cpu]`
extras bundle pulling in `default-stt` (faster-whisper → torch +
CUDA libs) and `default-tts` (kokoro → transformers + torch). For
Nimbus we only need `lite-stt` + `lite-tts`. Filed as VOICE-001.
Functional now, just bloated. Don't ship this image to Nimbus over a
residential pull — fix VOICE-001 first or build directly on the
target host.
