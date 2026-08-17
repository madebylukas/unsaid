# unsaid

> type without speaking.

![unsaid interface](docs/hero.png)

`unsaid` is a local, webcam-only visual speech demo for macOS. It records a short silent clip, runs an open-vocabulary Auto-AVSR model locally, and exposes several beam-search hypotheses instead of laundering uncertainty into one suspiciously confident sentence.

No microphone. No cloud inference. No three-word training ritual.

## What changed

The first prototype trained a tiny landmark classifier on words such as `pat`, `bat`, and `mat`. It was visually fun and statistically useless: with little data, it could collapse onto one class. This version replaces that toy model with a pretrained visual-speech system and keeps the browser responsible for the things it does well:

- live face and mouth tracking with MediaPipe;
- measured capture quality from frame rate, light, face angle, and mouth size;
- video-only recording;
- a legible transcript and N-best candidate list.

The Python side owns mouth-crop extraction, the visual transformer, and beam search.

## Run on a Mac

Requirements: an Apple Silicon Mac, Node 20+, Python 3.12 via [`uv`](https://docs.astral.sh/uv/), and roughly 3 GB free for dependencies and model files.

```bash
git clone https://github.com/madebylukas/unsaid.git
cd unsaid
npm install
uv python install cpython-3.12-macos-aarch64
npm run setup:vsr
```

Then use two terminals:

```bash
npm run vsr
```

```bash
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173), enable the camera, mouth a short phrase, and stop. The first read is slower because the model loads lazily.

The interface starts in a constrained six-phrase demo mode:

- `it's pretty cool you know`
- `hello how are you`
- `thank you very much`
- `i don't know`
- `open github`
- `send the message`

Mouth one line exactly. The raw visual beam is matched against this set and commits only when its match and separation clear the threshold; otherwise it says `not confident`. Switch to `open` in the interface for unrestricted decoding.

CPU is the reliable default because ESPnet's legacy beam search still mixes CPU tensors into MPS decoding. You can experiment with MPS, but it is not yet the honest default:

```bash
UNSAID_DEVICE=mps npm run vsr
```

## Architecture

```mermaid
flowchart LR
    A["webcam / video only"] --> B["live capture checks"]
    A --> C["short WebM clip"]
    C --> D["local face tracking + mouth crops"]
    D --> E["Auto-AVSR visual encoder"]
    E --> F["CTC / attention beam search"]
    G["local language model"] --> F
    F --> H["transcript + N-best candidates"]
```

The browser talks only to `127.0.0.1:8787`. Clips are written to a temporary local file for inference and deleted immediately afterward.

## Honest limits

Visual speech is underdetermined. `p`, `b`, and `m` can look the same because voicing and nasal airflow are not visible. A language model can rerank plausible readings; it cannot recover photons the camera never received.

This is a demo, not accessibility software and not a 99%-accurate Whisper replacement. The upstream checkpoint reports roughly 19% word error on the controlled LRS3 benchmark. Real webcams, unseen faces, bad light, facial hair, and casual mouthing are harder. The useful questions are whether the intended words survive in the top candidates, how often the system abstains, and how quickly personal corrections can improve it.

Read [RESEARCH.md](RESEARCH.md) for the evidence and [IDEATION.md](IDEATION.md) for the accuracy roadmap.

## Verification

```bash
npm test
npm run build
python3 -m py_compile backend/server.py
```

End-to-end browser validation on an M1 Pro used the built-in 1280×720 camera at 30 FPS. A 4.8-second cold clip decoded in 13.7 seconds; a 4.5-second warm clip decoded in 8.7 seconds. Both returned five distinct candidates. Treat those as one-machine measurements, not marketing scripture.

## Privacy

- The browser requests `audio: false`.
- Inference stays on the Mac.
- Temporary clips are deleted after each request.
- No accounts, analytics, or third-party API calls exist at runtime.

Privacy claims should survive `rg`, not merely a tasteful black interface.

## Upstream work

The local inference bridge adapts [Chaplin](https://github.com/amanvirparhar/chaplin) (MIT), which builds on [Auto-AVSR](https://github.com/mpc001/auto_avsr) (Apache-2.0). Model and language-model files retain their upstream terms. See [backend/NOTICE.md](backend/NOTICE.md).

## Notes

- [research review](RESEARCH.md)
- [accuracy roadmap](IDEATION.md)
- [visual system](DESIGN.md)
- [launch recipe](LAUNCH.md)

## License

[MIT](LICENSE), excluding third-party models and dependencies under their own terms.
