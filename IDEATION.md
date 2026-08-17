# Accuracy and product ideation

## Product thesis

The near-term product is not “Whisper without sound.” That promise invites an immediate and humiliating comparison against an information-richer channel.

The wedge is:

> **Private, personalised silent input for short, contextual communication.**

Examples: replying during a meeting, prompting an AI in public, controlling an app while audio is inappropriate, or selecting from contextually constrained actions.

The value meter is **accepted silent words per minute with zero edits**, not a flattering lab accuracy number.

## First-principles accuracy stack

### 1. Improve the information before enlarging the model

Bad framing cannot be gradient-descented into good evidence.

- Gate capture on mouth pixel width, pose, illumination and frame rate.
- Include a neutral pose before and after articulation.
- Preserve the full lower face as well as a tight mouth crop; jaw and cheek motion carry signal.
- Add lip pixels to landmarks. Landmarks provide geometry; pixels retain closure, teeth and tongue.
- Ask users to move closer instead of silently accepting garbage input.

### 2. Make time explicit

A lip shape is not a word. Motion is.

- Resample takes for the demo, then graduate to variable-length encoders.
- Model velocity and acceleration, not only positions.
- Use a 3D CNN or video transformer front end plus a Conformer/temporal transformer.
- Train with modest time warping, frame dropping and speed variation.
- Learn start/end detection separately from word identity.

### 3. Personalise cheaply

Cross-speaker generalisation is difficult because anatomy and articulation vary. The product can turn this weakness into an onboarding ritual.

- Start from a general pretrained visual encoder.
- Freeze most weights and train a small LoRA/prompt/adapter per user.
- Collect corrections as high-value labelled examples.
- Prefer active learning: ask for another take of the candidate pair the model confuses most.
- Maintain device/lighting profiles only if measurement proves they help.

### 4. Preserve the lattice

The visual model should emit phoneme, sub-word or word distributions. Keep the top candidates and their timing. Do not ask a language model to repair one already-broken sentence.

For every candidate sequence, retain:

- visual log probability;
- language/context log probability;
- personal vocabulary prior;
- quality score;
- model calibration temperature.

Fuse them with explicit weights and tune those weights on held-out users. This makes hallucination debuggable.

### 5. Use context without surrendering meaning

Useful context, in descending order of safety:

1. The words already dictated in the same sentence.
2. User-approved vocabulary, names and technical terms.
3. The active application's category: code editor, email, messaging, search.
4. Selected text or a small user-approved screen excerpt.
5. Broad language-model plausibility.

Context should rerank candidates present in the visual lattice. Generating an unrelated fluent sentence is not accuracy; it is fan fiction.

### 6. Abstain intelligently

There are two honest 99% claims:

- 99% accuracy on a small constrained vocabulary; or
- 99% precision on accepted outputs while declining ambiguous cases.

For the latter, expose a coverage curve. A system that is right 99% of the time but only answers 6% of the time is technically magnificent and practically a houseplant.

Use three interaction states:

- **commit:** one candidate dominates and quality is high;
- **suggest:** show two or three alternatives;
- **retry:** evidence is too weak or out of distribution.

## Model roadmap

### V0 — this repository

- Personal word classifier from mouth landmarks
- Fixed-length captures
- Curated local context prior
- Visible probability fusion
- Goal: prove the interaction and get real confusion data

### V1 — serious personalised commands

- Pretrained lip-pixel encoder based on LipLearner-style contrastive learning
- Landmark/pixel dual stream
- Visual keyword spotting
- Per-user linear head or adapter
- 30–100 commands
- Goal: greater than 95% top-1 on five personal examples per command across three sessions

### V2 — constrained dictation

- Phoneme/sub-word CTC or RNN-T output
- Beam search with an on-device language model
- Personal lexicon and app context
- Confidence-gated insertion
- Goal: short messages in a defined domain with a competitive zero-edit rate

### V3 — open vocabulary

- Auto-AVSR/VALLR-class visual encoder and decoder
- Core ML or MLX conversion for Apple Silicon
- Distillation, quantisation and streaming inference
- Optional EMG/contact sensing for the information that vision cannot recover
- Goal: honest WER and coverage benchmarks across unseen speakers, lighting and poses

## Experiments that matter

### A. Landmarks versus pixels

Train the same personal vocabulary with:

- landmarks only;
- 64×64 grayscale mouth crops only;
- both streams fused.

Measure top-1, top-3 and calibration. The expectation is that pixels help most on closure and tongue/teeth distinctions while landmarks stabilise pose and small data.

### B. More data or better pretraining?

Compare 1, 3, 5 and 10 personal takes with a random encoder versus a pretrained contrastive encoder. If five takes do not approach a plateau, onboarding is too expensive.

### C. Candidate preservation

Measure how often the correct word exists in top 3 even when top 1 is wrong. High top-3 recall validates context reranking; low top-3 recall means the visual encoder lost the word and no LLM should be invited to improvise.

### D. Context weight

Sweep the fusion weight. Report visual-only, context-only and fused scores. Test adversarial contexts where the visually correct word is semantically surprising.

### E. Cross-session decay

Train on Monday morning; test at night and again several days later under different light. Same-session accuracy is the friendliest possible exam and should not run the company.

## Metrics

- Top-1 and top-3 word accuracy
- Sentence WER when continuous decoding exists
- Zero-edit rate
- Expected calibration error
- Accepted-output precision versus coverage
- Median capture-to-insertion latency
- False activation rate per hour
- Accuracy across sessions, poses and lighting
- Onboarding minutes until useful

## Defensible moat

The interface can be cloned. The moat would be:

- consented paired silent-video and text data;
- longitudinal per-user adaptation data;
- an excellent visual activity detector;
- calibrated fusion of visual, lexical and application context;
- a low-friction correction loop that improves the model without becoming homework.

The brand can travel first. The dataset has to catch up quickly.
