"""Shared configuration for Qwen3-Coder fine-tuning pipeline."""

import os

# Nebius Token Factory API
NEBIUS_API_BASE = "https://api.tokenfactory.nebius.com/v1/"

# Base model to fine-tune
# NOTE: Verify this exact string against the Nebius model catalog.
# It may differ from the HuggingFace name (e.g. lowercase, no org prefix).
# To check, run: curl -H "Authorization: Bearer $NEBIUS_API_KEY" https://api.tokenfactory.nebius.com/v1/models
MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

# Polling interval in seconds for job status checks
POLL_INTERVAL = 15

# Random seed for reproducibility
SEED = 42
