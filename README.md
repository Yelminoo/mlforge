# MLForge 🖼️

**Image model fine-tuning framework — CLI + Web UI**

Fine-tune text-to-image and image-to-image models (FLUX, Stable Diffusion, ControlNet) with a single command or a browser UI. Supports LoRA, DreamBooth, full fine-tune, and textual inversion. Deploy to HuggingFace, Replicate, Modal, or RunPod.

---

## Install

```bash
git clone https://github.com/your-org/mlforge
cd mlforge
pip install -e .

# Optional extras
pip install -e ".[xformers]"   # faster attention
pip install -e ".[eval]"       # FID / CLIP / IS scoring
```

---

## CLI

### Interactive setup wizard
```bash
mlforge init
# → walks you through model, dataset, method, deploy target
# → saves forge.json
```

### Run full pipeline
```bash
mlforge run --config forge.json
```

### Individual stages
```bash
# List available base models
mlforge model list

# Fine-tune
mlforge train \
  --model black-forest-labs/FLUX.1-dev \
  --dataset laion/laion2B-en-aesthetic \
  --task t2i \
  --method lora \
  --epochs 10 \
  --lr 1e-4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --output-dir ./output

# Or from config
mlforge train --config forge.json

# Evaluate
mlforge eval --config forge.json

# Deploy
mlforge deploy --platform hf --repo-id your-org/my-model
mlforge deploy --platform replicate
mlforge deploy --platform modal
mlforge deploy --platform runpod

# Generate an image
mlforge generate \
  --prompt "a cinematic sunset over mountains" \
  --model-path ./output/final \
  --output result.png \
  --steps 30 \
  --guidance 7.5
```

### Launch web UI
```bash
mlforge serve --port 7860
# → open http://localhost:7860
```

---

## Web UI

The web UI mirrors every CLI option with live training logs streamed via SSE.

```
mlforge serve
```

Features:
- Model picker (FLUX.1, SD 3.5, ControlNet, IP-Adapter, …)
- Dataset loader (HF Hub · upload · URL)
- Preprocessing toggles (CLIP filter, aesthetic scoring, BLIP-2 auto-caption)
- Training config (LoRA / DreamBooth / full / textual inversion)
- Live training log + progress bar
- FID / CLIP / IS score dashboard
- One-click deploy with generated config

---

## Python API

```python
from core.pipeline import ForgeConfig, ModelConfig, DataConfig, TrainConfig, ForgePipeline

cfg = ForgeConfig(
    model=ModelConfig(
        model_id="black-forest-labs/FLUX.1-dev",
        task="t2i",
        torch_dtype="float16",
    ),
    data=DataConfig(
        dataset_id="laion/laion2B-en-aesthetic",
        max_samples=5000,
        clip_filter=True,
        aesthetic_filter=True,
    ),
    train=TrainConfig(
        method="lora",
        output_dir="./output",
        epochs=10,
        learning_rate=1e-4,
        lora_rank=16,
        lora_alpha=32,
    ),
)

pipe = ForgePipeline(cfg)

# Each stage yields StepResult objects
for result in pipe.load_model():
    print(result.status, result.message)

for result in pipe.load_dataset():
    print(result.status, result.message)

for result in pipe.train(progress_cb=lambda step, total, loss: print(f"{step}/{total} loss={loss:.4f}")):
    print(result.status, result.message)

for result in pipe.evaluate():
    print(result.status, result.message)

for result in pipe.deploy():
    print(result.status, result.message)

# Save / load config
cfg.save("forge.json")
cfg2 = ForgeConfig.load("forge.json")
```

---

## Supported models

| Model | Task | Size | Method |
|---|---|---|---|
| FLUX.1-dev | T2I | 12B | LoRA, full |
| FLUX.1-schnell | T2I | 12B | LoRA |
| SD 3.5 Large | T2I | 8B | LoRA, DreamBooth |
| PixArt-Σ | T2I | 600M | LoRA, full |
| SD 1.5 | T2I + I2I | 860M | All methods |
| InstructPix2Pix | I2I | 1B | LoRA, full |
| ControlNet v1.1 | I2I | 1.5B | LoRA |
| IP-Adapter | I2I | — | LoRA |

## Supported datasets

| Dataset | Task | Size |
|---|---|---|
| LAION-Aesthetics v2 | T2I | ~600K |
| DiffusionDB | T2I | 14M |
| JourneyDB | T2I | 4M |
| Conceptual Captions | T2I | 3M–12M |
| InstructPix2Pix | I2I | ~450K |
| MagicBrush | I2I | ~10K |
| MultiGen-20M | I2I | 20M |

## Deploy targets

| Platform | Type | Best for |
|---|---|---|
| HuggingFace Endpoints | Managed | Easiest path |
| Replicate | API | SaaS / sharing |
| Modal | Serverless GPU | Pay-per-generation |
| RunPod + FastAPI | Self-hosted | Full control |

---

## Project structure

```
mlforge/
├── cli/
│   └── mlforge.py        ← CLI entry point (argparse)
├── core/
│   └── pipeline.py       ← Pipeline engine (model, data, train, eval, deploy)
├── web/
│   ├── app.py            ← FastAPI server + SSE streaming
│   └── templates/
│       └── index.html    ← Single-page web UI
├── setup.py
└── README.md
```

---

## Requirements

- Python ≥ 3.10
- CUDA GPU recommended (≥ 16 GB VRAM for LoRA, ≥ 40 GB for full fine-tune)
- See `setup.py` for full dependency list
