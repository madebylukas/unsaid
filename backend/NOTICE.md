# Upstream attribution

The local inference bridge uses code and model interfaces from:

- [Chaplin](https://github.com/amanvirparhar/chaplin), MIT, copyright 2025 Amanvir Parhar.
- [Auto-AVSR](https://github.com/mpc001/auto_avsr), Apache-2.0, copyright 2023 Imperial College London / Pingchuan Ma.

The setup script checks out Chaplin at commit `7aee1f8fca776ce4f63690063310b53573b7d804` and downloads its linked Auto-AVSR visual and language-model checkpoints from Hugging Face. Those downloaded files are ignored by Git and retain their upstream terms.
