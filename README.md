# Fine-Tuning LLMs via Nebius Token Factory

Fine-tune large language models using LoRA or full fine-tuning through the [Nebius Token Factory](https://tokenfactory.nebius.com/) API. Supports 38+ models across Llama, Qwen, DeepSeek, and GPT-OSS families.

## Supported Models

### Fine-Tunable Models

Use `python create_job.py --list` to see all models, or `--search <term>` to filter.

| Family | Models | Params | Fine-Tuning | License |
|---|---|---|---|---|
| **Llama 3.1** | 8B, 8B-Instruct, 70B, 70B-Instruct | 8B–70B | LoRA + Full | Llama 3.1 Community |
| **Llama 3.2** | 1B, 1B-Instruct, 3B, 3B-Instruct | 1B–3B | LoRA + Full | Llama 3.2 Community |
| **Llama 3.3** | 70B-Instruct | 70B | LoRA + Full | Llama 3.3 Community |
| **Qwen3** | 0.6B–32B (dense + base variants) | 0.6B–32B | LoRA + Full | Apache 2.0 |
| **Qwen3 Coder** | 30B-A3B, 480B-A35B (MoE) | 3B–35B active | Full only | Apache 2.0 |
| **Qwen2.5** | 0.5B–72B (dense + coder + instruct) | 0.5B–72B | LoRA + Full | Apache 2.0 |
| **GPT-OSS** | 20B, 120B (Unsloth BF16) | 20B–120B | LoRA + Full | Apache 2.0 |
| **DeepSeek V3** | V3-0324, V3.1 (MoE) | 685B total | Full only | MIT |

**LoRA-deployable models** (serverless adapter deployment): `llama-3.1-8b-instruct`, `llama-3.3-70b-instruct`

### Inference-Only Models (not fine-tunable)

These models are available for inference through Nebius but cannot be fine-tuned:

| Alias | Model | Description | Pricing |
|---|---|---|---|
| `nemotron` | nvidia/nemotron-3-super-120b-a12b | 120B/12B active hybrid MoE, multi-agent & reasoning, 1M context | $0.30/1M in, $0.90/1M out |
| `kimi-k2` | moonshotai/Kimi-K2.5 | Native multimodal agentic model, ~15T token pretraining | $0.50/1M in, $2.50/1M out |

## How Nebius Token Factory Fine-Tuning Works

Nebius Token Factory provides an OpenAI-compatible API for fine-tuning LLMs. The process:

1. **Prepare datasets** — Create JSONL files containing chat-formatted training examples (system/user/assistant message triples).
2. **Upload datasets** — Send the JSONL files to Nebius via the Files API. Each upload returns a file ID.
3. **Submit a fine-tuning job** — Specify the base model, uploaded file IDs, and hyperparameters. Nebius runs training on their infrastructure — no GPUs needed on your end.
4. **Poll for completion** — The job runs asynchronously. Poll the Jobs API to track progress.
5. **Download checkpoints** — Once succeeded, download the adapter weights (LoRA) or full model checkpoint.

**LoRA** (Low-Rank Adaptation) inserts small trainable matrices into the model's attention layers instead of updating all parameters. This makes training fast and produces small adapter files (~100s of MB). Most models support LoRA. Some large MoE models (Qwen3 Coder, DeepSeek V3) only support full fine-tuning.

## Dataset Format

### Structure

Datasets are JSONL files (JSON Lines) — one valid JSON object per line, no pretty-printing, no trailing commas.

Each line follows this chat message schema:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

| Field | Required | Description |
|---|---|---|
| `messages` | Yes | Array of message objects (minimum 2) |
| `messages[].role` | Yes | One of `"system"`, `"user"`, or `"assistant"` |
| `messages[].content` | Yes | Non-empty string. Newlines in code must be escaped as `\n` within the JSON string |

**Rules:**
- Each line must contain at least one `"user"` message and one `"assistant"` message
- The `"system"` message is optional but recommended for consistency
- The `"assistant"` content is what the model learns to generate — keep it focused (code only, no markdown fences, no explanation text)
- All content strings must be non-empty

### Baseline Dataset

This project includes a minimal baseline dataset for pipeline validation:

| File | Lines | Purpose |
|---|---|---|
| `training.jsonl` | 10 examples | Python code-generation tasks (reverse string, is_prime, fibonacci, flatten list, find_max, word frequency, merge sorted lists, is_palindrome, remove duplicates, factorial) |
| `validation.jsonl` | 3 examples | Held-out tasks (GCD, balanced brackets, decimal-to-binary) — no overlap with training |

The validation set must contain tasks **not present** in the training set so the model's generalization can be measured.

### Creating Your Own Dataset

To build a production dataset:

1. **Format** — One JSON object per line, following the schema above. Use a consistent system message across all examples.
2. **Size** — More examples generally improve quality. The Nebius API requires that `batch_size × context_length ≥ 262,144` tokens per batch, so very small datasets need a larger batch size.
3. **Quality** — The assistant responses are your ground truth. Make sure they are correct, consistent in style, and represent exactly how you want the model to respond.
4. **Validation split** — Hold out 10–20% of examples for validation. Ensure no task overlap with training.
5. **Escaping** — Since each JSONL line is a single line of text, all newlines within code must be escaped as `\n` inside the JSON strings.
6. **Validate locally** — Run `validate_dataset.py` on your files before uploading to catch schema errors early.

## Project Structure

```
nebius-hackathon/
├── config.py                  # Model registry (fine-tune + inference), API URLs, hyperparameters
├── requirements.txt           # Python dependencies (openai SDK)
├── validate_dataset.py        # Local JSONL schema validation
├── upload_data.py             # Upload datasets to Nebius, saves file_ids.json
├── create_job.py              # Submit fine-tuning job (--model, --list, --search), saves job_id.json
├── poll_job.py                # Poll job status until completion
├── download_checkpoints.py    # Download adapter weights after job succeeds
├── training.jsonl             # Training dataset (10 examples)
├── validation.jsonl           # Validation dataset (3 examples)
├── file_ids.json              # [generated] Nebius file IDs from upload
├── job_id.json                # [generated] Nebius job ID and model from job creation
├── checkpoints/               # [generated] Downloaded adapter weights
└── .gitignore                 # Ignores generated files, .env, __pycache__, etc.
```

### File Descriptions

| File | Description |
|---|---|
| **config.py** | Central configuration. Contains `FINETUNE_MODELS` (38 models with params, ft_type, license), `INFERENCE_MODELS` (inference-only with pricing and regional endpoints), default hyperparameters, and helper functions (`get_api_base`, `get_finetune_model`, `list_models`). Add new models here. |
| **requirements.txt** | Lists `openai>=1.40.0` as the sole dependency. The Nebius Token Factory API is OpenAI-compatible, so the official OpenAI Python SDK is used as the client. |
| **validate_dataset.py** | Reads one or more JSONL files and checks every line for: valid JSON, presence of `messages` key, correct roles, non-empty content, and at least one user + assistant message. Prints per-file summaries and exits with code 1 if any errors are found. |
| **upload_data.py** | Uploads `training.jsonl` and `validation.jsonl` to Nebius using the Files API with `purpose="fine-tune"`. Saves file IDs to `file_ids.json`. |
| **create_job.py** | Submits a fine-tuning job. Flags: `--model` / `-m` (alias), `--list` / `-l` (show all models), `--search` (filter models), `--suffix` / `-s` (model name suffix). Automatically selects LoRA or full fine-tuning based on model capability. |
| **poll_job.py** | Polls the Jobs API every 15 seconds, printing timestamps and step progress. Handles Ctrl+C gracefully. Reads model from `job_id.json` to route to correct endpoint. |
| **download_checkpoints.py** | Verifies the job succeeded, downloads all checkpoint adapter files into `checkpoints/<checkpoint-id>/`. |

## Default Hyperparameters

Applied to all fine-tuning jobs. For full-only models, LoRA params are automatically removed.

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Samples per training step. Must satisfy `batch_size × context_length ≥ 262,144` |
| `learning_rate` | 1e-5 | Step size for optimizer updates |
| `n_epochs` | 3 | Number of passes over the full training set |
| `warmup_ratio` | 0.0 | Fraction of steps for learning rate warmup |
| `weight_decay` | 0.0 | L2 regularization coefficient |
| `lora` | true | Enable LoRA (auto-disabled for full-only models) |
| `lora_r` | 16 | LoRA rank — controls adapter capacity |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout applied to LoRA layers |
| `packing` | true | Pack multiple short examples into one sequence for efficiency |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `context_length` | 8192 | Maximum token length per training sequence |
| `seed` | 42 | Random seed for reproducibility |

## Prerequisites

- Python 3.10+
- A Nebius API key (sign up at [Nebius Token Factory](https://tokenfactory.nebius.com/))
- `uv` (recommended) or `pip` for dependency management

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd nebius-hackathon

# 2. Set your API key
export NEBIUS_API_KEY=<your-key>

# 3. Install dependencies
uv pip install -r requirements.txt

# 4. Browse available models
uv run python create_job.py --list
uv run python create_job.py --search llama

# 5. Validate datasets locally
uv run python validate_dataset.py training.jsonl validation.jsonl

# 6. Upload datasets to Nebius
uv run python upload_data.py

# 7. Submit the fine-tuning job (pick a model)
uv run python create_job.py --model qwen3-coder-480b      # default (full FT)
uv run python create_job.py --model llama-3.3-70b-instruct # LoRA
uv run python create_job.py --model qwen3-8b               # LoRA + full
uv run python create_job.py --model deepseek-v3.1          # full FT, US only

# 8. Monitor until completion (Ctrl+C is safe — re-run to resume)
uv run python poll_job.py

# 9. Download adapter checkpoints
uv run python download_checkpoints.py

# 10. Verify
ls -la checkpoints/
```

## Adding a New Model

1. Open `config.py`
2. Add to `FINETUNE_MODELS` (or `INFERENCE_MODELS` if inference-only):
   ```python
   "my-model": {
       "model_id": "org/model-name",
       "family": "Model Family",
       "params": "7B",
       "ft_type": "lora+full",       # or "full"
       "license": "Apache 2.0",
       "api_base": "https://...",     # optional, omit to use default endpoint
       "note": "US only",             # optional
   },
   ```
3. Run `uv run python create_job.py --model my-model`
