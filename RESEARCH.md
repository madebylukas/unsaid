# Research review

Research cut: **17 August 2026**. The aim was not to collect every lip-reading paper ever inflicted on a GPU. It was to decide what can make a credible Mac demo now, what should be borrowed later, and where the physics refuses negotiation.

## What exists

### Open-vocabulary visual speech recognition

**Auto-AVSR** provides reproducible audio and visual speech models. Its published open-source checkpoint reports about **20.3% word error rate (WER)** on LRS3; the multilingual successor reports **19.1% visual-only WER** with an approximately 891 MB checkpoint. It uses a visual front end plus sequence modelling and beam decoding. CPU inference is supported, but the stack and checkpoint are hefty for a link-clickable demo.

- [Auto-AVSR repository](https://github.com/mpc001/auto_avsr)
- [Visual Speech Recognition for Multiple Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages)

**AV-HuBERT** demonstrated that self-supervised audio-visual pretraining transfers strongly to lip reading. Its official repository supports video-only inference and beam search, but was archived in September 2024 and expects an older Fairseq/Python environment. It is research infrastructure, not a pleasant `npm install`.

- [Official AV-HuBERT repository](https://github.com/facebookresearch/av_hubert)

**VALLR** makes the architectural argument closest to this project's thesis: predict a compact phoneme sequence from video, then use a language model to reconstruct text. It reported **18.7% WER on LRS3**, or 17.5% in an additional configuration, while explicitly warning that directly bridging pixels to fluent text can hallucinate.

- [VALLR, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Thomas_VALLR_Visual_ASR_Language_Model_for_Lip_Reading_ICCV_2025_paper.html)

**Chaplin** already wraps an Auto-AVSR checkpoint, webcam capture and an Ollama cleanup pass into a local typing tool. It proves that open-vocabulary webcam inference is possible. Its trade-offs are an 891 MB-class model, Python/Torch/OpenCV/MediaPipe dependencies, a separate Qwen model, delayed clip inference, and an interface designed as a utility rather than an explanatory demo.

- [Chaplin repository](https://github.com/amanvirparhar/chaplin)

### Personalised and few-shot systems

**LipLearner** is the strongest precedent for this MVP shape. It used contrastive visual pretraining plus on-device customisation. The paper reports an F1 score of 0.8947 for 25 commands with one example in its model experiment; in the user study, 30-command accuracy increased from **81.7% with one example to 98.8% with five examples**. That is closed-vocabulary command recognition, not arbitrary transcription—but it is highly relevant product evidence.

- [LipLearner, CHI 2023](https://arxiv.org/abs/2302.05907)
- [LipLearner source](https://github.com/rkmtlab/LipLearner)

**Personalized Lip Reading** adapts both visual and language components to an individual and reports larger gains than vision-only adaptation. The key lesson is not merely “fine-tune on a face”; lexical habits are also personal.

- [Personalized Lip Reading, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33026)

### Browser tracking

Google's **MediaPipe Face Landmarker** exposes dense face landmarks in real time on the web. It is excellent for transparent live tracking and capture checks. Landmarks discard useful texture, tongue and tooth information, so they are a demo-friendly front end—not the accuracy ceiling.

- [MediaPipe Face Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js)

## The irreducible problem

Different phonemes can share nearly identical visible articulation. `p`, `b`, and `m` are canonical examples; papers call the resulting units **visemes** and the confusable words **homophenes**. Connected-speech context resolves many cases, but context is prior information, not newly recovered visual evidence.

This leads to three rules:

1. Preserve distributions or an N-best lattice; never collapse uncertainty early.
2. Measure calibration and top-k recall, not only top-1 accuracy.
3. Give the user a veto whenever the visual channel cannot distinguish meanings safely.

Sub-word visual attention work explicitly identifies `pa / ba / ma` ambiguity and finds sub-word units useful for modelling it.

- [Sub-word Level Lip Reading with Visual Attention, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Prajwal_Sub-Word_Level_Lip_Reading_With_Visual_Attention_CVPR_2022_paper.pdf)

## Why this implementation

| Route | First-run weight | Open vocabulary | Personal learning | Visual explainability | Mac demo friction |
|---|---:|---:|---:|---:|---:|
| Auto-AVSR/Chaplin | roughly 1 GB plus LLM | yes | no by default | medium | high |
| Landmark classifier | under 30 MB runtime assets | no | yes | excellent | low |
| Lip-pixel pretrained encoder | tens to hundreds of MB | commands | yes | good | medium |
| Camera plus EMG | hardware-dependent | eventually | yes | medium | very high |

The original landmark classifier won the first prototype because it could be understood and trained in minutes. It also collapsed onto one label with tiny personal datasets. This version therefore takes the heavier but more credible Auto-AVSR/Chaplin route: pretrained lip pixels, open vocabulary, local beam decoding, and visible alternatives. The next serious gain is a small personalised adapter, not another from-scratch classifier.

## Gaps worth attacking

- **Texture stream:** add a 64×64 mouth crop encoder alongside landmarks for teeth, tongue and lip closure cues.
- **Speaker adaptation:** learn a small per-user adapter rather than retraining the whole visual encoder.
- **Visual speech activity detection:** distinguish intentional mouthing from smiling, chewing and ordinary facial motion.
- **Sequence boundaries:** replace a fixed recording button with a confidence-gated visual keyword or deliberate mouth gesture.
- **Calibration:** temperature-scale probabilities on held-out personal takes.
- **Context:** replace the curated lexical prior with an on-device masked language model or N-gram model; keep its score separate.
- **Benchmark hygiene:** report user-independent and user-dependent results separately. Mixing them is how demos become folklore.

## Dataset caution

LRS2/LRS3 and VoxCeleb-derived work enabled the field, but datasets, pretrained checkpoints and redistributed videos can carry separate access and usage terms. This repository deliberately ships neither training video nor third-party VSR checkpoints. Before distributing a commercial model, audit every dataset and checkpoint licence rather than assuming the code licence blesses the weights.
