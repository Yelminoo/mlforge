#!/usr/bin/env python3
"""
MLForge CLI — fine-tune and deploy image generation models from your terminal.

Usage examples
--------------
  # Interactive setup wizard
  mlforge init

  # Full pipeline from a config file
  mlforge run --config forge.json

  # One-shot commands
  mlforge model list
  mlforge train --method lora --model black-forest-labs/FLUX.1-dev --dataset laion/laion2B-en-aesthetic
  mlforge eval --output ./output
  mlforge deploy --platform hf --repo-id your-org/my-model

  # Generate an image with your fine-tuned model
  mlforge generate --prompt "a sunset over mountains" --output result.png
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── pretty printing ────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
BLUE   = "\033[34m"
WHITE  = "\033[97m"

def c(color, text):    return f"{color}{text}{RESET}"
def ok(msg):           print(f"  {c(GREEN,'✔')} {msg}")
def warn(msg):         print(f"  {c(YELLOW,'⚠')} {msg}")
def err(msg):          print(f"  {c(RED,'✖')} {msg}", file=sys.stderr)
def info(msg):         print(f"  {c(CYAN,'›')} {msg}")
def head(msg):         print(f"\n{c(BOLD+WHITE, msg)}")
def dim(msg):          print(f"  {c(DIM, msg)}")
def hr():              print(f"  {c(DIM, '─' * 56)}")

def step_result(r):
    """Print a StepResult from the pipeline."""
    dispatch = {"ok": ok, "warn": warn, "error": err, "info": info}
    dispatch.get(r.status, info)(r.message)

def progress_bar(step, total, loss, width=40):
    filled = int(width * step / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = int(100 * step / max(total, 1))
    print(f"\r  {c(CYAN, bar)} {pct:3d}%  step {step}/{total}  loss={loss:.4f}", end="", flush=True)
    if step >= total:
        print()

# ── banner ─────────────────────────────────────────────────────────────────────

BANNER = f"""
{c(BOLD+CYAN,  '  ███╗   ███╗██╗      ███████╗ ██████╗ ██████╗  ██████╗ ███████╗')}
{c(BOLD+CYAN,  '  ████╗ ████║██║      ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝')}
{c(BOLD+WHITE, '  ██╔████╔██║██║      █████╗  ██║   ██║██████╔╝██║  ███╗█████╗  ')}
{c(BOLD+WHITE, '  ██║╚██╔╝██║██║      ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  ')}
{c(BOLD+BLUE,  '  ██║ ╚═╝ ██║███████╗ ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗')}
{c(BOLD+BLUE,  '  ╚═╝     ╚═╝╚══════╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝')}
{c(DIM,        '  Image model fine-tuning · CLI + Web · v1.0.0')}
"""

# ── model registry ──────────────────────────────────────────────────────────────

MODELS = {
    "t2i": [
        {"id": "black-forest-labs/FLUX.1-dev",                        "name": "FLUX.1-dev",     "tags": ["12B", "Apache 2.0", "best quality"]},
        {"id": "stabilityai/stable-diffusion-3.5-large",              "name": "SD 3.5 Large",   "tags": ["8B",  "open",       "versatile"]},
        {"id": "black-forest-labs/FLUX.1-schnell",                    "name": "FLUX.1-schnell", "tags": ["12B", "Apache 2.0", "fast"]},
        {"id": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",             "name": "PixArt-Σ",       "tags": ["600M","open",       "efficient"]},
    ],
    "i2i": [
        {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5",         "name": "SD 1.5",          "tags": ["860M","open",       "most resources"]},
        {"id": "timbrooks/instruct-pix2pix",                          "name": "InstructPix2Pix", "tags": ["1B",  "open",       "text-guided edits"]},
        {"id": "lllyasviel/ControlNet-v1-1",                          "name": "ControlNet v1.1", "tags": ["1.5B","open",       "structured gen"]},
        {"id": "h94/IP-Adapter",                                      "name": "IP-Adapter",      "tags": ["open","image prompts"]},
    ],
}

DATASETS = {
    "t2i": [
        {"id": "laion/laion2B-en-aesthetic",             "name": "LAION-Aesthetics v2",  "size": "~600K"},
        {"id": "poloclub/diffusiondb",                   "name": "DiffusionDB",           "size": "14M"},
        {"id": "JourneyDB/JourneyDB",                    "name": "JourneyDB",             "size": "4M"},
        {"id": "google-research-datasets/conceptual_captions", "name": "Conceptual Captions", "size": "3M"},
    ],
    "i2i": [
        {"id": "timbrooks/instructpix2pix-clip-filtered","name": "InstructPix2Pix",       "size": "~450K"},
        {"id": "osunlp/MagicBrush",                      "name": "MagicBrush",            "size": "~10K"},
        {"id": "EPFL-VILAB/MultiGen-20M",                "name": "MultiGen-20M",          "size": "20M"},
    ],
}


# ── subcommand: init ────────────────────────────────────────────────────────────

def cmd_init(args):
    print(BANNER)
    head("Interactive setup wizard")
    hr()

    # Task
    print(f"\n  {c(BOLD,'Task type:')}")
    print("    1) Text → Image")
    print("    2) Image → Image")
    print("    3) Both")
    t = input(f"\n  {c(CYAN,'Choose [1-3] (default 1): ')}").strip() or "1"
    task = {"1": "t2i", "2": "i2i", "3": "both"}.get(t, "t2i")

    # Model
    model_list = MODELS.get(task, MODELS["t2i"])
    print(f"\n  {c(BOLD,'Base model:')}")
    for i, m in enumerate(model_list, 1):
        tags = "  ".join(c(DIM, f"[{tag}]") for tag in m["tags"])
        print(f"    {i}) {c(WHITE, m['name'])} {tags}")
    print(f"    {len(model_list)+1}) Custom HuggingFace model ID")
    mi = input(f"\n  {c(CYAN,'Choose [1-{len(model_list)+1}] (default 1): ')}").strip() or "1"
    if mi.isdigit() and 1 <= int(mi) <= len(model_list):
        model_id = model_list[int(mi)-1]["id"]
    else:
        model_id = input(f"  {c(CYAN,'HuggingFace model ID: ')}").strip()

    # Dataset
    ds_list = DATASETS.get(task, DATASETS["t2i"])
    print(f"\n  {c(BOLD,'Dataset:')}")
    for i, d in enumerate(ds_list, 1):
        print(f"    {i}) {c(WHITE, d['name'])} {c(DIM, d['size'])}")
    print(f"    {len(ds_list)+1}) Custom dataset ID or local path")
    di = input(f"\n  {c(CYAN,'Choose [1-{len(ds_list)+1}] (default 1): ')}").strip() or "1"
    if di.isdigit() and 1 <= int(di) <= len(ds_list):
        dataset_id = ds_list[int(di)-1]["id"]
    else:
        dataset_id = input(f"  {c(CYAN,'Dataset ID or path: ')}").strip()

    # Method
    print(f"\n  {c(BOLD,'Training method:')}")
    print("    1) LoRA         — fast, low VRAM, ~50-200 MB output")
    print("    2) DreamBooth   — learn a concept (style/subject)")
    print("    3) Full fine-tune — highest quality, needs ≥40 GB VRAM")
    print("    4) Textual inversion — tiny file, limited flexibility")
    mm = input(f"\n  {c(CYAN,'Choose [1-4] (default 1): ')}").strip() or "1"
    method = {"1": "lora", "2": "dreambooth", "3": "full", "4": "textinv"}.get(mm, "lora")

    # Deploy target
    print(f"\n  {c(BOLD,'Deploy to:')}")
    print("    1) HuggingFace Endpoints  — easiest")
    print("    2) Replicate              — versioned API")
    print("    3) Modal                  — serverless GPU")
    print("    4) RunPod                 — self-hosted FastAPI")
    dp = input(f"\n  {c(CYAN,'Choose [1-4] (default 1): ')}").strip() or "1"
    platform = {"1": "hf", "2": "replicate", "3": "modal", "4": "runpod"}.get(dp, "hf")

    output_dir = input(f"\n  {c(CYAN,'Output directory [./output]: ')}").strip() or "./output"

    # Build and save config
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.pipeline import ForgeConfig, ModelConfig, DataConfig, TrainConfig, DeployConfig
    cfg = ForgeConfig(
        model=ModelConfig(model_id=model_id, task=task),
        data=DataConfig(dataset_id=dataset_id),
        train=TrainConfig(method=method, output_dir=output_dir),
        deploy=DeployConfig(platform=platform),
    )
    out_path = args.output or "forge.json"
    cfg.save(out_path)
    hr()
    ok(f"Config saved → {c(WHITE, out_path)}")
    info(f"Next: {c(CYAN, f'mlforge run --config {out_path}')}")


# ── subcommand: model ───────────────────────────────────────────────────────────

def cmd_model(args):
    if args.model_cmd == "list":
        head("Available base models")
        for task, models in MODELS.items():
            print(f"\n  {c(BOLD, task.upper())}")
            hr()
            for m in models:
                tags = "  ".join(c(DIM, f"[{t}]") for t in m["tags"])
                print(f"  {c(WHITE, m['name']):<28} {tags}")
                dim(m["id"])


# ── subcommand: train ────────────────────────────────────────────────────────────

def cmd_train(args):
    head("MLForge — Training")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.pipeline import ForgeConfig, ModelConfig, DataConfig, TrainConfig, ForgePipeline

    if args.config:
        cfg = ForgeConfig.load(args.config)
    else:
        cfg = ForgeConfig(
            model=ModelConfig(model_id=args.model, task=args.task),
            data=DataConfig(dataset_id=args.dataset, max_samples=args.max_samples),
            train=TrainConfig(
                method=args.method,
                output_dir=args.output_dir,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
            ),
        )

    pipe = ForgePipeline(cfg)
    hr()
    head("1 / 3  Loading model")
    for r in pipe.load_model(): step_result(r)
    head("2 / 3  Loading dataset")
    for r in pipe.load_dataset(): step_result(r)
    head("3 / 3  Fine-tuning")
    hr()
    for r in pipe.train(progress_cb=progress_bar): step_result(r)
    hr()
    ok("Training complete!")
    cfg_hint = args.config or "forge.json"
    info(f"Run evaluation: {c(CYAN, 'mlforge eval --config ' + cfg_hint)}")


# ── subcommand: eval ─────────────────────────────────────────────────────────────

def cmd_eval(args):
    head("MLForge — Evaluation")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.pipeline import ForgeConfig, ForgePipeline
    cfg = ForgeConfig.load(args.config) if args.config else ForgeConfig()
    pipe = ForgePipeline(cfg)
    hr()
    for r in pipe.evaluate(): step_result(r)
    hr()
    ok("Evaluation complete!")
    info(f"Deploy: {c(CYAN, 'mlforge deploy --platform hf')}")


# ── subcommand: deploy ───────────────────────────────────────────────────────────

def cmd_deploy(args):
    head("MLForge — Deploy")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.pipeline import ForgeConfig, DeployConfig, ForgePipeline
    if args.config:
        cfg = ForgeConfig.load(args.config)
    else:
        cfg = ForgeConfig()
    if args.platform:   cfg.deploy.platform = args.platform
    if args.repo_id:    cfg.deploy.repo_id  = args.repo_id
    if args.hf_token:   cfg.deploy.hf_token = args.hf_token
    pipe = ForgePipeline(cfg)
    hr()
    for r in pipe.deploy(): step_result(r)
    hr()
    ok("Deploy step complete!")


# ── subcommand: run (full pipeline) ─────────────────────────────────────────────

def cmd_run(args):
    print(BANNER)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.pipeline import ForgeConfig, ForgePipeline
    cfg = ForgeConfig.load(args.config)

    pipe = ForgePipeline(cfg)
    stages = [
        ("Load model",   pipe.load_model),
        ("Load dataset", pipe.load_dataset),
        ("Train",        lambda: pipe.train(progress_cb=progress_bar)),
        ("Evaluate",     pipe.evaluate),
        ("Deploy",       pipe.deploy),
    ]

    for i, (name, fn) in enumerate(stages, 1):
        head(f"{i} / {len(stages)}  {name}")
        hr()
        for r in fn(): step_result(r)
        print()

    hr()
    ok(f"Pipeline complete!  Model ready at {c(WHITE, cfg.train.output_dir)}")


# ── subcommand: generate ─────────────────────────────────────────────────────────

def cmd_generate(args):
    head("MLForge — Generate")
    info(f"Prompt: {c(WHITE, args.prompt)}")
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        pipe = AutoPipelineForText2Image.from_pretrained(
            args.model_path, torch_dtype=torch.float16
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        info("Generating …")
        image = pipe(args.prompt, num_inference_steps=args.steps,
                     guidance_scale=args.guidance).images[0]
        out = args.output or "output.png"
        image.save(out)
        ok(f"Saved → {c(WHITE, out)}")
    except ImportError:
        warn("diffusers/torch not installed — dry-run mode")
        ok(f"[dry-run] Would save image to {args.output or 'output.png'}")
    except Exception as e:
        err(str(e))


# ── subcommand: serve (web UI) ────────────────────────────────────────────────────

def cmd_serve(args):
    head("MLForge — Web UI")
    info(f"Starting server on http://0.0.0.0:{args.port}")
    try:
        import uvicorn
        web_dir = str(Path(__file__).parent.parent / "web" / "app.py")
        uvicorn.run("web.app:app", host="0.0.0.0", port=args.port, reload=args.reload)
    except ImportError:
        err("fastapi/uvicorn not installed. Run: pip install fastapi uvicorn")


# ── build parser ─────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="mlforge",
        description="Image model fine-tuning framework — CLI + Web",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    # init
    pi = sub.add_parser("init", help="Interactive setup wizard")
    pi.add_argument("-o", "--output", default="forge.json", help="Config output path")

    # model
    pm = sub.add_parser("model", help="Model registry commands")
    pm.add_subparsers(dest="model_cmd").add_parser("list", help="List available models")

    # train
    pt = sub.add_parser("train", help="Run fine-tuning")
    pt.add_argument("--config",      help="Load from forge.json")
    pt.add_argument("--model",       default="black-forest-labs/FLUX.1-dev")
    pt.add_argument("--dataset",     default="laion/laion2B-en-aesthetic")
    pt.add_argument("--task",        default="t2i", choices=["t2i","i2i","both"])
    pt.add_argument("--method",      default="lora", choices=["lora","dreambooth","full","textinv"])
    pt.add_argument("--output-dir",  default="./output")
    pt.add_argument("--lr",          type=float, default=1e-4)
    pt.add_argument("--batch-size",  type=int,   default=4)
    pt.add_argument("--epochs",      type=int,   default=10)
    pt.add_argument("--lora-rank",   type=int,   default=16)
    pt.add_argument("--lora-alpha",  type=int,   default=32)
    pt.add_argument("--max-samples", type=int,   default=5000)

    # eval
    pe = sub.add_parser("eval", help="Evaluate fine-tuned model")
    pe.add_argument("--config", help="forge.json path")

    # deploy
    pd = sub.add_parser("deploy", help="Deploy model")
    pd.add_argument("--config",   help="forge.json path")
    pd.add_argument("--platform", choices=["hf","replicate","modal","runpod"])
    pd.add_argument("--repo-id",  help="HuggingFace repo id")
    pd.add_argument("--hf-token", help="HuggingFace API token")

    # run (full pipeline)
    pr = sub.add_parser("run", help="Run full pipeline from config")
    pr.add_argument("--config", required=True, help="forge.json path")

    # generate
    pg = sub.add_parser("generate", help="Generate an image")
    pg.add_argument("--prompt",     required=True)
    pg.add_argument("--model-path", default="./output/final")
    pg.add_argument("--output",     default="output.png")
    pg.add_argument("--steps",      type=int,   default=30)
    pg.add_argument("--guidance",   type=float, default=7.5)

    # serve
    ps = sub.add_parser("serve", help="Launch web UI")
    ps.add_argument("--port",   type=int, default=7860)
    ps.add_argument("--reload", action="store_true")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "init":     cmd_init,
        "model":    cmd_model,
        "train":    cmd_train,
        "eval":     cmd_eval,
        "deploy":   cmd_deploy,
        "run":      cmd_run,
        "generate": cmd_generate,
        "serve":    cmd_serve,
    }

    fn = dispatch.get(args.cmd)
    if fn:
        fn(args)
    else:
        print(BANNER)
        parser.print_help()


if __name__ == "__main__":
    main()
