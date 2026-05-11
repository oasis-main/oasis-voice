# Queue — oasis-voice (Division Level)

Items are listed in priority order within each section.
Item IDs: `VOICE-<3-digit-number>` — assigned sequentially, never reused.

This queue tracks **oasis-voice** (Python, sister to oasis-claw). For the
openclaw integration side of voice work see oasis-claw's `.swarm/queue.md`
under the `CLAW-XXX` namespace.

---

## Active

- [ ] [VOICE-001] [OPEN] Slim the `oasis-voice:cpu` image from 6.37GB to ~500MB
      priority: high | project: cpu-image | division: VOICE
      assignees: TBD
      depends: none
      notes: The `oasis-voice:cpu` image as of `436a64f` weighs in at
             **6.37GB** — twelve times the ≲500MB target stated in
             `docker/Dockerfile.cpu`. Root cause is the `[cpu]`
             convenience bundle in `pyproject.toml` line 33:

               cpu = ["oasis-voice[lite-stt,default-stt,stream-light,lite-tts,default-tts]"]

             It pulls in:
               - `default-stt` → `faster-whisper>=1.1` → ctranslate2 +
                 torch (~2GB of CUDA libs come along even on a "CPU"
                 install because torch's manylinux wheel bundles them)
               - `default-tts` → `kokoro>=0.7` → transformers + torch
                 (same CUDA-library issue)
               - `stream-light` → `sherpa-onnx>=1.12` (legit, modest)
               - `lite-stt` → `useful-moonshine-onnx` (legit, ONNX
                 runtime only)
               - `lite-tts` → `piper-tts` (legit, no torch)

             For Nimbus's first-call MVP we ONLY use lite-stt + lite-tts.
             The other tiers are aspirational and shouldn't be in the
             CPU bundle that ships to the laptop / cloud-CPU instance.

             Proposed fix:
               1. Split the convenience bundles:
                    cpu-lite  = ["oasis-voice[lite-stt,lite-tts]"]
                    cpu-full  = ["oasis-voice[lite-stt,default-stt,
                                              stream-light,lite-tts,
                                              default-tts]"]
                    cpu       = alias to cpu-lite for now (rename
                                later if the breaking-change is
                                tolerable)
               2. Update `docker/Dockerfile.cpu` to install
                  `[cpu-lite]` not `[cpu]`. The Dockerfile already
                  documents "Only lite/default/stream-light tiers" but
                  the bundle definition contradicted that.
               3. Add an integration test that asserts the built image
                  is < 1GB. Easy gate: `docker image inspect ...
                  --format '{{.Size}}' | awk '{ exit ($1 > 1073741824) }'`.

             Why this matters NOW: oasis-claw plans to ship this image
             to Nimbus (an Apple-silicon Mac on a residential
             connection). 6.37GB takes ~15min to pull on a typical home
             upload; 500MB takes ~1min. Also affects oasis-cloud
             (CLAW-013) ECS pull time.

             Won't touch any oasis-claw integration — the
             MediaUnderstandingProvider in
             `extensions/oasis-voice/media-understanding-provider.ts`
             only POSTs to `/v1/stt/transcribe`, so as long as that
             endpoint still works, image-slim is invisible to consumers.

             Acceptance:
               - `docker images oasis-voice:cpu --format '{{.Size}}'`
                 reports < 1GB.
               - Existing pytest suite (37 tests, 1 skipped on host
                 without ffmpeg) still passes inside the new container.
               - `POST /v1/stt/transcribe` with a small Telegram-shaped
                 .opus blob returns a transcript inside the slimmed
                 image (proves Moonshine + ffmpeg paths still work).
               - oasis-claw's E2E "Telegram voice in" smoke still works
                 against the slimmed image (no integration regression).

---

## Backlog

(empty)

---

## Done (recent)

(none yet — repo just stood up its dev branch)
