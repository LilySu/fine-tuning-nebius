"""Shared configuration for Nebius Token Factory fine-tuning and inference pipeline."""

import os

# Nebius Token Factory API — default fine-tuning endpoint
NEBIUS_API_BASE = "https://api.tokenfactory.nebius.com/v1/"

# Polling interval in seconds for job status checks
POLL_INTERVAL = 15

# Random seed for reproducibility
SEED = 42

# ---------------------------------------------------------------------------
# Fine-tunable models — these can be used with create_job.py
# ---------------------------------------------------------------------------
# "ft_type" indicates supported fine-tuning: "lora+full", or "full"
# "lora_deployable" means the LoRA adapter can be deployed as serverless
FINETUNE_MODELS = {
    # --- Meta Llama 3.1 ---
    "llama-3.1-8b-instruct": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "family": "Llama 3.1",
        "params": "8B",
        "ft_type": "lora+full",
        "lora_deployable": True,
        "license": "Llama 3.1 Community",
    },
    "llama-3.1-8b": {
        "model_id": "meta-llama/Meta-Llama-3.1-8B",
        "family": "Llama 3.1",
        "params": "8B",
        "ft_type": "lora+full",
        "license": "Llama 3.1 Community",
    },
    "llama-3.1-70b-instruct": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "family": "Llama 3.1",
        "params": "70B",
        "ft_type": "lora+full",
        "license": "Llama 3.1 Community",
    },
    "llama-3.1-70b": {
        "model_id": "meta-llama/Llama-3.1-70B",
        "family": "Llama 3.1",
        "params": "70B",
        "ft_type": "lora+full",
        "license": "Llama 3.1 Community",
    },
    # --- Meta Llama 3.2 ---
    "llama-3.2-1b-instruct": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "family": "Llama 3.2",
        "params": "1B",
        "ft_type": "lora+full",
        "license": "Llama 3.2 Community",
    },
    "llama-3.2-1b": {
        "model_id": "meta-llama/Llama-3.2-1B",
        "family": "Llama 3.2",
        "params": "1B",
        "ft_type": "lora+full",
        "license": "Llama 3.2 Community",
    },
    "llama-3.2-3b-instruct": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "family": "Llama 3.2",
        "params": "3B",
        "ft_type": "lora+full",
        "license": "Llama 3.2 Community",
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B",
        "family": "Llama 3.2",
        "params": "3B",
        "ft_type": "lora+full",
        "license": "Llama 3.2 Community",
    },
    # --- Meta Llama 3.3 ---
    "llama-3.3-70b-instruct": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "family": "Llama 3.3",
        "params": "70B",
        "ft_type": "lora+full",
        "lora_deployable": True,
        "license": "Llama 3.3 Community",
    },
    # --- Qwen3 dense + base ---
    "qwen3-32b": {
        "model_id": "Qwen/Qwen3-32B",
        "family": "Qwen3",
        "params": "32B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-14b": {
        "model_id": "Qwen/Qwen3-14B",
        "family": "Qwen3",
        "params": "14B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-14b-base": {
        "model_id": "Qwen/Qwen3-14B-Base",
        "family": "Qwen3",
        "params": "14B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "family": "Qwen3",
        "params": "8B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-8b-base": {
        "model_id": "Qwen/Qwen3-8B-Base",
        "family": "Qwen3",
        "params": "8B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-4b": {
        "model_id": "Qwen/Qwen3-4B",
        "family": "Qwen3",
        "params": "4B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-4b-base": {
        "model_id": "Qwen/Qwen3-4B-Base",
        "family": "Qwen3",
        "params": "4B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "family": "Qwen3",
        "params": "1.7B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-1.7b-base": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "family": "Qwen3",
        "params": "1.7B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-0.6b": {
        "model_id": "Qwen/Qwen3-0.6B",
        "family": "Qwen3",
        "params": "0.6B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen3-0.6b-base": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "family": "Qwen3",
        "params": "0.6B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    # --- Qwen3 Coder (MoE — full fine-tuning only) ---
    "qwen3-coder-30b": {
        "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "family": "Qwen3 Coder",
        "params": "30B total / 3B active (MoE)",
        "ft_type": "full",
        "license": "Apache 2.0",
    },
    "qwen3-coder-480b": {
        "model_id": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "family": "Qwen3 Coder",
        "params": "480B total / 35B active (MoE)",
        "ft_type": "full",
        "license": "Apache 2.0",
    },
    # --- Qwen2.5 dense + coder ---
    "qwen2.5-0.5b": {
        "model_id": "Qwen/Qwen2.5-0.5B",
        "family": "Qwen2.5",
        "params": "0.5B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-0.5b-instruct": {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "family": "Qwen2.5",
        "params": "0.5B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B",
        "family": "Qwen2.5",
        "params": "7B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-7b-instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "family": "Qwen2.5",
        "params": "7B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-14b": {
        "model_id": "Qwen/Qwen2.5-14B",
        "family": "Qwen2.5",
        "params": "14B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-14b-instruct": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "family": "Qwen2.5",
        "params": "14B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-32b": {
        "model_id": "Qwen/Qwen2.5-32B",
        "family": "Qwen2.5",
        "params": "32B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-32b-instruct": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "family": "Qwen2.5",
        "params": "32B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-72b": {
        "model_id": "Qwen/Qwen2.5-72B",
        "family": "Qwen2.5",
        "params": "72B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-72b-instruct": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "family": "Qwen2.5",
        "params": "72B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-coder-32b": {
        "model_id": "Qwen/Qwen2.5-Coder-32B",
        "family": "Qwen2.5 Coder",
        "params": "32B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "qwen2.5-coder-32b-instruct": {
        "model_id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "family": "Qwen2.5 Coder",
        "params": "32B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    # --- GPT-OSS (Unsloth) ---
    "gpt-oss-20b": {
        "model_id": "unsloth/gpt-oss-20b-BF16",
        "family": "GPT-OSS",
        "params": "20B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    "gpt-oss-120b": {
        "model_id": "unsloth/gpt-oss-120b-BF16",
        "family": "GPT-OSS",
        "params": "120B",
        "ft_type": "lora+full",
        "license": "Apache 2.0",
    },
    # --- DeepSeek V3 (US data centers only) ---
    "deepseek-v3-0324": {
        "model_id": "deepseek-ai/DeepSeek-V3-0324",
        "family": "DeepSeek V3",
        "params": "685B total (MoE)",
        "ft_type": "full",
        "license": "MIT",
        "note": "US data centers only",
    },
    "deepseek-v3.1": {
        "model_id": "deepseek-ai/DeepSeek-V3.1",
        "family": "DeepSeek V3",
        "params": "685B total (MoE)",
        "ft_type": "full",
        "license": "MIT",
        "note": "US data centers only",
    },
}

# ---------------------------------------------------------------------------
# Inference-only models — NOT fine-tunable, use with inference scripts only
# ---------------------------------------------------------------------------
INFERENCE_MODELS = {
    "nemotron": {
        "model_id": "nvidia/nemotron-3-super-120b-a12b",
        "description": "120B total / 12B active hybrid MoE, multi-agent & reasoning, 1M context",
        "api_base": "https://api.tokenfactory.us-central1.nebius.com/v1/",
        "pricing": "$0.30/1M in, $0.90/1M out",
    },
    "kimi-k2": {
        "model_id": "moonshotai/Kimi-K2.5",
        "description": "Native multimodal agentic model, ~15T token pretraining",
        "api_base": "https://api.tokenfactory.me-west1.nebius.com/v1/",
        "pricing": "$0.50/1M in, $2.50/1M out",
    },
}

# Default model alias for fine-tuning (used when --model is not specified)
DEFAULT_MODEL = "qwen3-coder-480b"

# Default hyperparameters — applied to all fine-tuning jobs unless overridden
DEFAULT_HYPERPARAMETERS = {
    "batch_size": 32,
    "learning_rate": 1e-5,
    "n_epochs": 3,
    "warmup_ratio": 0.0,
    "weight_decay": 0.0,
    "lora": True,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "packing": True,
    "max_grad_norm": 1.0,
    "context_length": 8192,
}


def get_api_base(model_alias=None):
    """Return the API base URL for a model, falling back to the default."""
    if model_alias and model_alias in FINETUNE_MODELS:
        return FINETUNE_MODELS[model_alias].get("api_base", NEBIUS_API_BASE)
    if model_alias and model_alias in INFERENCE_MODELS:
        return INFERENCE_MODELS[model_alias].get("api_base", NEBIUS_API_BASE)
    return NEBIUS_API_BASE


def get_finetune_model(alias):
    """Look up a fine-tunable model by alias. Returns (model_id, config) or raises KeyError."""
    if alias in INFERENCE_MODELS:
        raise KeyError(
            f"'{alias}' is an inference-only model and cannot be fine-tuned. "
            f"Use --list to see fine-tunable models."
        )
    if alias not in FINETUNE_MODELS:
        raise KeyError(f"Unknown model alias '{alias}'. Use --list to see available models.")
    return FINETUNE_MODELS[alias]


def list_models(filter_text=None):
    """Print available models, optionally filtered by a search string."""
    print("=" * 80)
    print("FINE-TUNABLE MODELS")
    print("=" * 80)
    print(f"{'Alias':<32} {'Params':<28} {'FT Type':<14} {'License'}")
    print("-" * 80)
    for alias, cfg in FINETUNE_MODELS.items():
        if filter_text and filter_text.lower() not in alias and filter_text.lower() not in cfg["model_id"].lower():
            continue
        note = f"  ({cfg['note']})" if cfg.get("note") else ""
        print(f"{alias:<32} {cfg['params']:<28} {cfg['ft_type']:<14} {cfg['license']}{note}")

    print()
    print("=" * 80)
    print("INFERENCE-ONLY MODELS (not fine-tunable)")
    print("=" * 80)
    print(f"{'Alias':<16} {'Model ID':<42} {'Pricing'}")
    print("-" * 80)
    for alias, cfg in INFERENCE_MODELS.items():
        if filter_text and filter_text.lower() not in alias and filter_text.lower() not in cfg["model_id"].lower():
            continue
        print(f"{alias:<16} {cfg['model_id']:<42} {cfg['pricing']}")
