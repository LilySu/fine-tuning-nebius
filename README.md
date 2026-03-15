# Qwen3-Coder Fine-Tuning via Nebius Token Factory

Fine-tuning **Qwen3-Coder-480B-A35B-Instruct** using LoRA adapters through the Nebius Token Factory API.

## Model

| Detail | Value |
|---|---|
| **Model** | Qwen/Qwen3-Coder-480B-A35B-Instruct |
| **Total Parameters** | 480 billion |
| **Active Parameters** | 35 billion (Mixture-of-Experts architecture) |
| **Model Family** | Qwen3-Coder |
| **Specialization** | Code generation and understanding |

Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) model — while it has 480B total parameters, only 35B are activated per forward pass, making it efficient for its size. The model is instruction-tuned for code-related tasks including generation, completion, debugging, and explanation.

## How Nebius Token Factory Fine-Tuning Works

[Nebius Token Factory](https://tokenfactory.nebius.com/) provides an OpenAI-compatible API for fine-tuning large language models. The process works as follows:

1. **Prepare datasets** — Create JSONL files containing chat-formatted training examples (system/user/assistant message triples).
2. **Upload datasets** — Send the JSONL files to Nebius via the Files API. Each upload returns a file ID.
3. **Submit a fine-tuning job** — Specify the base model, uploaded file IDs, and hyperparameters. Nebius runs LoRA (Low-Rank Adaptation) training on their infrastructure — you don't need your own GPUs.
4. **Poll for completion** — The job runs asynchronously. Poll the Jobs API to track progress until it reaches a terminal state (succeeded, failed, or cancelled).
5. **Download checkpoints** — Once succeeded, download the LoRA adapter weights. These are small files (~100s of MB) that layer on top of the base model at inference time.

LoRA fine-tuning modifies only a small number of low-rank matrices inserted into the model's attention layers, rather than updating all 480B parameters. This makes training feasible and fast while still adapting the model's behavior to your dataset.

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
├── config.py                  # Shared configuration (API URL, model name, constants)
├── requirements.txt           # Python dependencies (openai SDK)
├── validate_dataset.py        # Local JSONL schema validation
├── upload_data.py             # Upload datasets to Nebius, saves file_ids.json
├── create_job.py              # Submit LoRA fine-tuning job, saves job_id.json
├── poll_job.py                # Poll job status until completion
├── download_checkpoints.py    # Download adapter weights after job succeeds
├── training.jsonl             # Training dataset (10 examples)
├── validation.jsonl           # Validation dataset (3 examples)
├── file_ids.json              # [generated] Nebius file IDs from upload
├── job_id.json                # [generated] Nebius job ID from job creation
├── checkpoints/               # [generated] Downloaded adapter weights
└── .gitignore                 # Ignores generated files, .env, __pycache__, etc.
```

### File Descriptions

| File | Description |
|---|---|
| **config.py** | Central configuration. Contains the Nebius API base URL (`https://api.tokenfactory.nebius.com/v1/`), model identifier, polling interval (15s), and random seed (42). Edit this file to change the target model or tuning constants. |
| **requirements.txt** | Lists `openai>=1.40.0` as the sole dependency. The Nebius Token Factory API is OpenAI-compatible, so the official OpenAI Python SDK is used as the client. |
| **validate_dataset.py** | Reads one or more JSONL files and checks every line for: valid JSON, presence of `messages` key, correct roles, non-empty content, and at least one user + assistant message. Prints per-file summaries and exits with code 1 if any errors are found. Run this before uploading to catch problems locally. |
| **upload_data.py** | Uploads `training.jsonl` and `validation.jsonl` to Nebius using the Files API with `purpose="fine-tune"`. Prints the returned file IDs and saves them to `file_ids.json` for use by subsequent scripts. |
| **create_job.py** | Reads file IDs from `file_ids.json` and submits a LoRA fine-tuning job with configured hyperparameters. Saves the returned job ID to `job_id.json`. |
| **poll_job.py** | Reads the job ID from `job_id.json` and polls the Jobs API every 15 seconds, printing timestamps and step progress. Exits when the job reaches a terminal state. Handles Ctrl+C gracefully. |
| **download_checkpoints.py** | Verifies the job succeeded, lists all checkpoints, and downloads each checkpoint's adapter files into `checkpoints/<checkpoint-id>/`. Prints a summary of files downloaded. |

## Hyperparameters

The fine-tuning job in `create_job.py` uses these settings:

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 32 | Samples per training step. Must satisfy `batch_size × context_length ≥ 262,144` |
| `learning_rate` | 1e-5 | Step size for optimizer updates |
| `n_epochs` | 3 | Number of passes over the full training set |
| `warmup_ratio` | 0.0 | Fraction of steps for learning rate warmup |
| `weight_decay` | 0.0 | L2 regularization coefficient |
| `lora` | true | Enable LoRA (Low-Rank Adaptation) |
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

# 4. Validate datasets locally
uv run python validate_dataset.py training.jsonl validation.jsonl

# 5. Upload datasets to Nebius
uv run python upload_data.py

# 6. Submit the fine-tuning job
uv run python create_job.py

# 7. Monitor until completion (Ctrl+C is safe — re-run to resume)
uv run python poll_job.py

# 8. Download adapter checkpoints
uv run python download_checkpoints.py

# 9. Verify
ls -la checkpoints/
```
