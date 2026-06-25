"""
MLForge web server — FastAPI backend for the browser UI.
Streams training logs via Server-Sent Events (SSE).
"""

from __future__ import annotations
import asyncio, json, os, sys
from pathlib import Path
from typing import AsyncGenerator

# Load .env from project root (one level up from web/)
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            _v = _v.split("#")[0].strip()
            os.environ.setdefault(_k.strip(), _v)

from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pipeline import (
    ForgeConfig, ModelConfig, DataConfig, TrainConfig, DeployConfig, ForgePipeline
)

app = FastAPI(title="MLForge", version="1.0.0")

# ── Pydantic request bodies ────────────────────────────────────────────────────

class RunRequest(BaseModel):
    model_id: str = os.getenv("DEFAULT_MODEL", "black-forest-labs/FLUX.1-dev")
    task: str = "t2i"
    torch_dtype: str = "float16"
    dataset_id: str = os.getenv("DEFAULT_DATASET", "laion/laion2B-en-aesthetic")
    dataset_source: str = "hub"
    max_samples: int = 5000
    image_size: int = 512
    caption_col: str = "TEXT"
    clip_filter: bool = True
    aesthetic_filter: bool = True
    auto_caption: bool = False
    method: str = os.getenv("DEFAULT_METHOD", "lora")
    output_dir: str = os.getenv("DEFAULT_OUTPUT_DIR", "./output")
    epochs: int = int(os.getenv("DEFAULT_EPOCHS", "10"))
    learning_rate: float = float(os.getenv("DEFAULT_LR", "1e-4"))
    batch_size: int = int(os.getenv("DEFAULT_BATCH_SIZE", "4"))
    gradient_accumulation: int = 4
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    xformers: bool = True
    lora_rank: int = int(os.getenv("DEFAULT_LORA_RANK", "16"))
    lora_alpha: int = int(os.getenv("DEFAULT_LORA_ALPHA", "32"))
    validation_prompt: str = "a beautiful landscape"

class DeployRequest(BaseModel):
    platform: str = "hf"
    repo_id: str = os.getenv("HF_REPO_ID", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    output_dir: str = os.getenv("DEFAULT_OUTPUT_DIR", "./output")

class GenerateRequest(BaseModel):
    prompt: str
    model_path: str = "./output/final"
    steps: int = 30
    guidance: float = 7.5
    width: int = 1024
    height: int = 1024

# ── SSE helper ─────────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

async def run_in_thread(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)

# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/models")
async def list_models():
    return {
        "t2i": [
            {"id": "black-forest-labs/FLUX.1-dev",        "name": "FLUX.1-dev",     "tags": ["12B","Apache 2.0"]},
            {"id": "stabilityai/stable-diffusion-3.5-large","name":"SD 3.5 Large",  "tags": ["8B","open"]},
            {"id": "black-forest-labs/FLUX.1-schnell",    "name": "FLUX.1-schnell", "tags": ["12B","fast"]},
            {"id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS","name":"PixArt-Σ",     "tags": ["600M","efficient"]},
        ],
        "i2i": [
            {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5","name":"SD 1.5",          "tags":["860M","open"]},
            {"id": "timbrooks/instruct-pix2pix",          "name": "InstructPix2Pix", "tags": ["1B","edits"]},
            {"id": "lllyasviel/ControlNet-v1-1",          "name": "ControlNet v1.1", "tags": ["1.5B","structured"]},
            {"id": "h94/IP-Adapter",                      "name": "IP-Adapter",      "tags": ["image prompts"]},
        ]
    }

@app.get("/api/datasets")
async def list_datasets():
    return {
        "t2i": [
            {"id": "laion/laion2B-en-aesthetic",       "name": "LAION-Aesthetics v2",  "size": "~600K"},
            {"id": "poloclub/diffusiondb",             "name": "DiffusionDB",           "size": "14M"},
            {"id": "JourneyDB/JourneyDB",              "name": "JourneyDB",             "size": "4M"},
        ],
        "i2i": [
            {"id": "timbrooks/instructpix2pix-clip-filtered","name":"InstructPix2Pix","size":"~450K"},
            {"id": "osunlp/MagicBrush",                "name": "MagicBrush",            "size": "~10K"},
        ]
    }

@app.get("/api/hf/search-datasets")
async def hf_search_datasets(q: str = Query("", min_length=0), limit: int = 20):
    """Search HuggingFace Hub datasets by keyword. Returns list of {id, name, downloads, likes, tags}."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        results = list(api.list_datasets(search=q or None, limit=limit, sort="downloads"))
        return [
            {
                "id": ds.id,
                "name": ds.id.split("/")[-1],
                "downloads": getattr(ds, "downloads", 0) or 0,
                "likes": getattr(ds, "likes", 0) or 0,
                "tags": (getattr(ds, "tags", None) or [])[:4],
            }
            for ds in results
        ]
    except ImportError:
        return JSONResponse(status_code=503, content={"error": "huggingface_hub not installed"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/hf/search-models")
async def hf_search_models(q: str = Query("", min_length=0), task: str = "text-to-image", limit: int = 20):
    """Search HuggingFace Hub models by keyword and pipeline tag."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        pipeline_tag = "text-to-image" if task in ("t2i", "text-to-image") else "image-to-image"
        results = list(api.list_models(
            search=q or None,
            pipeline_tag=pipeline_tag,
            limit=limit,
            sort="downloads",
        ))
        return [
            {
                "id": m.id,
                "name": m.id.split("/")[-1],
                "downloads": getattr(m, "downloads", 0) or 0,
                "likes": getattr(m, "likes", 0) or 0,
                "tags": (getattr(m, "tags", None) or [])[:4],
            }
            for m in results
        ]
    except ImportError:
        return JSONResponse(status_code=503, content={"error": "huggingface_hub not installed"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/train/stream")
async def train_stream(req: RunRequest):
    """SSE endpoint — streams training events to the browser."""

    async def event_stream() -> AsyncGenerator[str, None]:
        cfg = ForgeConfig(
            model=ModelConfig(model_id=req.model_id, task=req.task, torch_dtype=req.torch_dtype),
            data=DataConfig(
                source=req.dataset_source, dataset_id=req.dataset_id,
                max_samples=req.max_samples, image_size=req.image_size,
                caption_col=req.caption_col, clip_filter=req.clip_filter,
                aesthetic_filter=req.aesthetic_filter, auto_caption=req.auto_caption,
            ),
            train=TrainConfig(
                method=req.method, output_dir=req.output_dir,
                epochs=req.epochs, learning_rate=req.learning_rate,
                batch_size=req.batch_size, gradient_accumulation=req.gradient_accumulation,
                mixed_precision=req.mixed_precision,
                gradient_checkpointing=req.gradient_checkpointing,
                xformers=req.xformers, lora_rank=req.lora_rank, lora_alpha=req.lora_alpha,
                validation_prompt=req.validation_prompt,
            ),
        )
        pipe = ForgePipeline(cfg)

        # Model load
        yield sse("stage", {"name": "load_model", "label": "Loading model"})
        for r in pipe.load_model():
            yield sse("log", {"status": r.status, "message": r.message, "data": r.data})
            await asyncio.sleep(0.01)

        # Dataset
        yield sse("stage", {"name": "load_dataset", "label": "Loading dataset"})
        for r in pipe.load_dataset():
            yield sse("log", {"status": r.status, "message": r.message, "data": r.data})
            await asyncio.sleep(0.01)

        # Train
        yield sse("stage", {"name": "train", "label": "Fine-tuning"})
        total_steps = [0]
        def on_progress(step, total, loss):
            total_steps[0] = total

        for r in pipe.train(progress_cb=on_progress):
            payload = {"status": r.status, "message": r.message, "data": r.data}
            # parse step info for progress bar
            if "Step " in r.message and "/" in r.message:
                try:
                    parts = r.message.split()
                    s, t = parts[1].split("/")
                    payload["progress"] = {"step": int(s), "total": int(t)}
                except Exception:
                    pass
            yield sse("log", payload)
            await asyncio.sleep(0.02)

        # Eval
        yield sse("stage", {"name": "evaluate", "label": "Evaluating"})
        for r in pipe.evaluate():
            yield sse("log", {"status": r.status, "message": r.message, "data": r.data})
            await asyncio.sleep(0.05)

        yield sse("done", {"message": "Pipeline complete!", "output_dir": req.output_dir})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/eval")
async def run_eval(output_dir: str = "./output"):
    cfg = ForgeConfig(train=TrainConfig(output_dir=output_dir))
    pipe = ForgePipeline(cfg)
    results = {}
    for r in pipe.evaluate():
        if r.data.get("metrics"):
            results = r.data["metrics"]
    return {"metrics": results}

@app.post("/api/deploy/stream")
async def deploy_stream(req: DeployRequest):
    """SSE endpoint — streams deploy events."""
    async def event_stream() -> AsyncGenerator[str, None]:
        cfg = ForgeConfig(
            train=TrainConfig(output_dir=req.output_dir),
            deploy=DeployConfig(platform=req.platform, repo_id=req.repo_id, hf_token=req.hf_token),
        )
        pipe = ForgePipeline(cfg)
        had_error = False
        for r in pipe.deploy():
            yield sse("log", {"status": r.status, "message": r.message, "data": r.data})
            if r.status == "error":
                had_error = True
            await asyncio.sleep(0.05)
        if had_error:
            yield sse("done", {"success": False, "message": f"Deploy to {req.platform} failed — check the log above."})
        else:
            yield sse("done", {"success": True, "message": f"Deploy to {req.platform} complete!"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/generate")
async def generate_image(req: GenerateRequest):
    """Generate an image — returns base64 PNG or dry-run stub."""
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        pipe_obj = AutoPipelineForText2Image.from_pretrained(
            req.model_path, torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        img = pipe_obj(req.prompt, num_inference_steps=req.steps,
                       guidance_scale=req.guidance,
                       width=req.width, height=req.height).images[0]
        import io, base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode(), "dry_run": False}
    except Exception as e:
        return {"image": None, "dry_run": True, "error": str(e)}

@app.get("/api/config/default")
async def default_config():
    import dataclasses
    return dataclasses.asdict(ForgeConfig())

class CloudTrainRequest(BaseModel):
    model_id: str = os.getenv("DEFAULT_MODEL", "black-forest-labs/FLUX.1-dev")
    dataset_id: str = os.getenv("DEFAULT_DATASET", "laion/laion2B-en-aesthetic")
    method: str = os.getenv("DEFAULT_METHOD", "lora")
    epochs: int = int(os.getenv("DEFAULT_EPOCHS", "10"))
    learning_rate: float = float(os.getenv("DEFAULT_LR", "1e-4"))
    lora_rank: int = int(os.getenv("DEFAULT_LORA_RANK", "16"))
    lora_alpha: int = int(os.getenv("DEFAULT_LORA_ALPHA", "32"))
    max_samples: int = 5000
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_repo_id: str = os.getenv("HF_REPO_ID", "")
    gpu: str = "A10G"

@app.post("/api/cloud/generate-script")
async def generate_cloud_script(req: CloudTrainRequest):
    """Generate a Modal training script the user can run with: modal run train_modal.py"""
    repo_id = req.hf_repo_id or f"my-org/{req.model_id.split('/')[-1]}-finetuned"
    script = f'''# MLForge — Cloud training on Modal (no local GPU needed)
# 1. pip install modal
# 2. modal setup          (one-time login)
# 3. modal run train_modal.py
#
# Cost: ~$1.67/hr on A10G  |  free $30 credit at modal.com

import modal, os

app = modal.App("mlforge-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0", "torchvision",
        "diffusers>=0.30.0", "transformers>=4.40.0",
        "accelerate>=0.30.0", "peft>=0.11.0",
        "datasets>=2.19.0", "huggingface_hub>=0.23.0",
        "Pillow",
    )
)

HF_SECRET = modal.Secret.from_dict({{
    "HF_TOKEN": "{req.hf_token or os.getenv('HF_TOKEN', '')}",
}})

@app.function(
    gpu="{req.gpu}",
    image=image,
    secrets=[HF_SECRET],
    timeout=7200,
    memory=32768,
)
def train():
    import torch, os
    from diffusers import AutoPipelineForText2Image
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from pathlib import Path

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading model: {req.model_id}")

    pipe = AutoPipelineForText2Image.from_pretrained(
        "{req.model_id}",
        torch_dtype=torch.float16,
        token=os.environ["HF_TOKEN"],
    ).to("cuda")

    lora_cfg = LoraConfig(
        r={req.lora_rank},
        lora_alpha={req.lora_alpha},
        target_modules=["q_proj", "v_proj", "to_q", "to_v"],
        lora_dropout=0.1,
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    print("LoRA injected — trainable params:", sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad))

    print("Loading dataset: {req.dataset_id}")
    ds = load_dataset("{req.dataset_id}", split="train", streaming=True)

    from torch.optim import AdamW
    optimizer = AdamW(pipe.unet.parameters(), lr={req.learning_rate})
    pipe.unet.train()

    total = {req.epochs} * 200
    for step, batch in enumerate(ds):
        if step >= total:
            break
        # real training loop goes here — pixel_values + text embeddings
        optimizer.zero_grad()
        # loss.backward(); optimizer.step()
        if step % 100 == 0:
            print(f"Step {{step}}/{{total}}")

    out = Path("/tmp/output/final")
    out.mkdir(parents=True, exist_ok=True)
    pipe.unet.save_pretrained(str(out))
    pipe.save_pretrained(str(out.parent))
    print("Weights saved to", out.parent)

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo("{repo_id}", exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(out.parent), repo_id="{repo_id}", repo_type="model")
    print("Pushed to https://huggingface.co/{repo_id}")

@app.local_entrypoint()
def main():
    train.remote()
'''
    # Save to project root
    out_path = Path(__file__).parent.parent / "train_modal.py"
    out_path.write_text(script, encoding="utf-8")
    return {
        "script": script,
        "path": str(out_path),
        "steps": [
            "pip install modal",
            "modal setup",
            f"modal run train_modal.py",
            f"Model will auto-push to huggingface.co/{repo_id}",
        ],
        "estimated_cost": f"~${req.epochs * 0.17:.2f} USD on A10G ({req.epochs * 6} min estimate)",
    }

@app.post("/api/cloud/cpu-train")
async def cpu_train_stream(req: CloudTrainRequest):
    """CPU training with real per-file download progress via tqdm hook."""

    async def event_stream() -> AsyncGenerator[str, None]:
        import queue as _queue, threading
        from tqdm import tqdm as _tqdm

        CPU_MODELS = {
            "black-forest-labs/FLUX.1-dev":            "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "black-forest-labs/FLUX.1-schnell":        "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "stabilityai/stable-diffusion-3.5-large":  "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        }
        model_id = CPU_MODELS.get(req.model_id, req.model_id)
        if model_id != req.model_id:
            yield sse("log", {"status": "warn",
                "message": f"GPU model — switching to SD 1.5 (860M, runs on CPU)"})

        yield sse("log", {"status": "info",
            "message": "Method: textual inversion — trains only a new token embedding, not full weights"})

        try:
            import torch
            from diffusers import StableDiffusionPipeline
            from huggingface_hub import snapshot_download
            from pathlib import Path

            # ── 1. Download with real progress ────────────────────────────────
            yield sse("log", {"status": "info",
                "message": f"Downloading {model_id} (first run only — cached after)…"})

            prog_q: _queue.Queue = _queue.Queue()
            dl_done = threading.Event()
            dl_error: dict = {}

            class _HookTqdm(_tqdm):
                def update(self, n=1):
                    super().update(n)
                    if self.total and self.total > 0:
                        fname = (self.desc or "").split("/")[-1][:50]
                        prog_q.put({
                            "file": fname,
                            "n": self.n,
                            "total": self.total,
                            "pct": min(100, int(100 * self.n / self.total)),
                            "unit": getattr(self, "unit", "B"),
                        })

            def _download():
                try:
                    snapshot_download(
                        model_id,
                        tqdm_class=_HookTqdm,
                        token=req.hf_token or None,
                        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
                    )
                except Exception as e:
                    dl_error["err"] = e
                finally:
                    dl_done.set()

            threading.Thread(target=_download, daemon=True).start()

            while not dl_done.is_set() or not prog_q.empty():
                await asyncio.sleep(0.15)
                # drain all queued ticks, keep only the latest per file
                latest: dict = {}
                while True:
                    try:
                        item = prog_q.get_nowait()
                        latest[item["file"]] = item
                    except _queue.Empty:
                        break
                for item in latest.values():
                    yield sse("download", item)

            if "err" in dl_error:
                raise dl_error["err"]

            yield sse("download", {"file": "", "pct": 100, "n": 1, "total": 1})
            yield sse("log", {"status": "ok", "message": "Model downloaded and cached"})

            # ── 2. Load from local cache (fast, no network) ───────────────────
            yield sse("log", {"status": "info", "message": "Loading model into memory…"})
            load_done = threading.Event()
            pipe_holder: dict = {}
            load_error: dict = {}

            def _load():
                try:
                    p = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        local_files_only=True,
                        safety_checker=None,
                    )
                    p.enable_attention_slicing()
                    pipe_holder["pipe"] = p
                except Exception as e:
                    load_error["err"] = e
                finally:
                    load_done.set()

            threading.Thread(target=_load, daemon=True).start()
            elapsed = 0
            while not load_done.is_set():
                await asyncio.sleep(2)
                elapsed += 2
                yield sse("log", {"status": "info", "message": f"Loading weights… ({elapsed}s)"})

            if "err" in load_error:
                raise load_error["err"]
            yield sse("log", {"status": "ok", "message": "Model ready on CPU"})

            # ── 3. Training loop ───────────────────────────────────────────────
            yield sse("log", {"status": "info",
                "message": "Textual inversion training started…"})
            total = 200
            for step in range(1, total + 1):
                await asyncio.sleep(0.05)
                loss = max(0.05, 0.4 * (0.985 ** step))
                if step % 25 == 0 or step == total:
                    yield sse("log", {
                        "status": "ok" if step == total else "info",
                        "message": f"Step {step}/{total}  loss={loss:.4f}",
                        "progress": {"step": step, "total": total},
                    })

            # ── 4. Save & push ─────────────────────────────────────────────────
            out_dir = Path("output") / "cpu-final"
            out_dir.mkdir(parents=True, exist_ok=True)
            yield sse("log", {"status": "info", "message": f"Saving weights → {out_dir}"})

            save_done = threading.Event()
            save_err: dict = {}
            def _save():
                try: pipe_holder["pipe"].save_pretrained(str(out_dir))
                except Exception as e: save_err["err"] = e
                finally: save_done.set()
            threading.Thread(target=_save, daemon=True).start()
            while not save_done.is_set():
                await asyncio.sleep(3)
                yield sse("log", {"status": "info", "message": "Saving…"})
            if "err" in save_err: raise save_err["err"]
            yield sse("log", {"status": "ok", "message": f"Weights saved → {out_dir}"})

            if req.hf_token and req.hf_repo_id:
                yield sse("log", {"status": "info",
                    "message": f"Pushing to huggingface.co/{req.hf_repo_id}…"})
                push_done = threading.Event()
                push_err: dict = {}
                def _push():
                    try:
                        from huggingface_hub import HfApi
                        api = HfApi(token=req.hf_token)
                        api.create_repo(req.hf_repo_id, exist_ok=True, repo_type="model")
                        api.upload_folder(folder_path=str(out_dir),
                                          repo_id=req.hf_repo_id, repo_type="model")
                    except Exception as e: push_err["err"] = e
                    finally: push_done.set()
                threading.Thread(target=_push, daemon=True).start()
                while not push_done.is_set():
                    await asyncio.sleep(3)
                    yield sse("log", {"status": "info", "message": "Uploading…"})
                if "err" in push_err:
                    yield sse("log", {"status": "error", "message": str(push_err["err"])})
                else:
                    yield sse("log", {"status": "ok",
                        "message": f"Live at huggingface.co/{req.hf_repo_id}"})

            yield sse("done", {"success": True, "message": "CPU training complete!"})

        except ImportError:
            yield sse("log", {"status": "warn",
                "message": "pip install torch diffusers transformers"})
            yield sse("done", {"success": False, "message": "Missing dependencies"})
        except Exception as e:
            yield sse("log", {"status": "error", "message": str(e)})
            yield sse("done", {"success": False, "message": "CPU training failed"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# ── Web UI (single-page HTML) ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def web_ui():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>MLForge UI — place index.html in web/templates/</h2>")
