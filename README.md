# Secure SemCom Attack & Defense Framework

This repository contains a semantic communication (SemCom) security framework:
- Encoder–semantic bottleneck–decoder pipeline
- Three attack types:
  - A1: input-level adversarial attack (PGD / FGSM)
  - A2: semantic-latent attack (PGD on bottleneck)
  - A3: task-logit tampering
- Plain-text SemCom evaluation (no ZKP/FP layer yet)

## Quick Start

```bash
pip install -r requirements.txt
python run_semcom.py

