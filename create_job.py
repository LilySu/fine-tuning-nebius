"""Submit a LoRA fine-tuning job to Nebius Token Factory."""

import argparse
import json
import os
import sys

from openai import OpenAI

from config import (
    DEFAULT_HYPERPARAMETERS,
    DEFAULT_MODEL,
    FINETUNE_MODELS,
    NEBIUS_API_BASE,
    SEED,
    get_api_base,
    get_finetune_model,
    list_models,
)


def main():
    parser = argparse.ArgumentParser(
        description="Submit a fine-tuning job to Nebius Token Factory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python create_job.py --list\n"
               "  python create_job.py --search qwen3\n"
               "  python create_job.py --model llama-3.3-70b-instruct\n"
               "  python create_job.py --model qwen3-8b --suffix my-experiment-v2\n",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model alias to fine-tune (default: {DEFAULT_MODEL}). Use --list to see all options.",
    )
    parser.add_argument(
        "--suffix", "-s",
        default="baseline-v1",
        help="Suffix for the fine-tuned model name (default: baseline-v1)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        dest="list_models",
        help="List all available models and exit",
    )
    parser.add_argument(
        "--search",
        help="Search models by name (e.g., --search qwen3, --search 70b)",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    if args.search:
        list_models(filter_text=args.search)
        sys.exit(0)

    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY environment variable is not set.")
        sys.exit(1)

    if not os.path.exists("file_ids.json"):
        print("ERROR: file_ids.json not found. Run upload_data.py first.")
        sys.exit(1)

    with open("file_ids.json") as f:
        file_ids = json.load(f)

    training_file_id = file_ids["training_file_id"]
    validation_file_id = file_ids["validation_file_id"]

    try:
        model_config = get_finetune_model(args.model)
    except KeyError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    model_id = model_config["model_id"]
    ft_type = model_config["ft_type"]
    api_base = get_api_base(args.model)

    # Use default hyperparameters; disable LoRA if model only supports full fine-tuning
    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    if ft_type == "full":
        hyperparameters["lora"] = False
        del hyperparameters["lora_r"]
        del hyperparameters["lora_alpha"]
        del hyperparameters["lora_dropout"]

    lora_label = "LoRA" if hyperparameters.get("lora") else "Full"
    print(f"Model:    {args.model} ({model_id})")
    print(f"Params:   {model_config['params']}")
    print(f"FT Type:  {lora_label} (supported: {ft_type})")
    print(f"Endpoint: {api_base}")
    print(f"Suffix:   {args.suffix}\n")

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    try:
        job = client.fine_tuning.jobs.create(
            model=model_id,
            training_file=training_file_id,
            validation_file=validation_file_id,
            suffix=args.suffix,
            hyperparameters=hyperparameters,
            seed=SEED,
        )
    except Exception as e:
        print(f"ERROR creating fine-tuning job: {e}")
        sys.exit(1)

    print("Job created successfully.")
    print(f"Job ID:  {job.id}")
    print(f"Status:  {job.status}")
    print(f"Model:   {job.model}")

    with open("job_id.json", "w") as f:
        json.dump({"job_id": job.id, "model": args.model}, f, indent=2)

    print("\nJob ID saved to job_id.json")


if __name__ == "__main__":
    main()
