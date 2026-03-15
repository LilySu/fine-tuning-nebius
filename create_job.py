"""Submit a LoRA fine-tuning job to Nebius Token Factory."""

import json
import os
import sys

from openai import OpenAI

from config import MODEL_NAME, NEBIUS_API_BASE, SEED


def main():
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

    client = OpenAI(
        base_url=NEBIUS_API_BASE,
        api_key=api_key,
    )

    try:
        job = client.fine_tuning.jobs.create(
            model=MODEL_NAME,
            training_file=training_file_id,
            validation_file=validation_file_id,
            suffix="baseline-v1",
            hyperparameters={
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
            },
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
        json.dump({"job_id": job.id}, f, indent=2)

    print("\nJob ID saved to job_id.json")


if __name__ == "__main__":
    main()
