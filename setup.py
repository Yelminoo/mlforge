"""MLForge — Image model fine-tuning framework: CLI + Web UI"""
from setuptools import setup, find_packages

setup(
    name="mlforge",
    version="1.0.0",
    description="Image model fine-tuning framework with CLI and Web UI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "diffusers>=0.30.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "peft>=0.11.0",
        "datasets>=2.19.0",
        "huggingface_hub>=0.23.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "Pillow>=10.0.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "xformers": ["xformers>=0.0.26"],
        "eval": ["torch-fidelity", "clip-score"],
        "dev": ["pytest", "black", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "mlforge=cli.mlforge:main",
        ]
    },
)
