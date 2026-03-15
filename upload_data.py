"""Upload training and validation JSONL files to Nebius Token Factory."""

import json
import os
import sys

from openai import OpenAI

from config import NEBIUS_API_BASE


def main():
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY environment variable is not set.")
        sys.exit(1)

    client = OpenAI(
        base_url=NEBIUS_API_BASE,
        api_key=api_key,
    )

    try:
        with open("training.jsonl", "rb") as f:
            training_file = client.files.create(file=f, purpose="fine-tune")
        print(f"Training file ID:   {training_file.id}")
    except Exception as e:
        print(f"ERROR uploading training.jsonl: {e}")
        sys.exit(1)

    try:
        with open("validation.jsonl", "rb") as f:
            validation_file = client.files.create(file=f, purpose="fine-tune")
        print(f"Validation file ID: {validation_file.id}")
    except Exception as e:
        print(f"ERROR uploading validation.jsonl: {e}")
        sys.exit(1)

    file_ids = {
        "training_file_id": training_file.id,
        "validation_file_id": validation_file.id,
    }

    with open("file_ids.json", "w") as f:
        json.dump(file_ids, f, indent=2)

    print(f"\nFile IDs saved to file_ids.json")


if __name__ == "__main__":
    main()
