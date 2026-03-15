"""Download adapter checkpoints from a completed Nebius fine-tuning job."""

import json
import os
import sys

from openai import OpenAI

from config import NEBIUS_API_BASE, get_api_base


def main():
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        print("ERROR: NEBIUS_API_KEY environment variable is not set.")
        sys.exit(1)

    if not os.path.exists("job_id.json"):
        print("ERROR: job_id.json not found. Run create_job.py first.")
        sys.exit(1)

    with open("job_id.json") as f:
        job_data = json.load(f)
    job_id = job_data["job_id"]
    model_alias = job_data.get("model")
    api_base = get_api_base(model_alias)

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )

    job = client.fine_tuning.jobs.retrieve(job_id)
    if job.status != "succeeded":
        print(f"Job status is '{job.status}', not 'succeeded'. Cannot download checkpoints.")
        sys.exit(1)

    checkpoints = client.fine_tuning.jobs.checkpoints.list(job_id).data

    if not checkpoints:
        print("No checkpoints found for this job.")
        sys.exit(1)

    total_files = 0

    for checkpoint in checkpoints:
        checkpoint_dir = os.path.join("checkpoints", checkpoint.id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"\nCheckpoint: {checkpoint.id} (step {checkpoint.step_number})")

        for file_id in checkpoint.result_files:
            try:
                file_obj = client.files.retrieve(file_id)
                filename = os.path.basename(file_obj.filename)
                output_path = os.path.join(checkpoint_dir, filename)

                file_content = client.files.content(file_id)
                file_content.write_to_file(output_path)
                print(f"  Saved: {output_path}")
                total_files += 1
            except Exception as e:
                print(f"  ERROR downloading {file_id}: {e}")

    print(f"\n=== Download Summary ===")
    print(f"Total checkpoints: {len(checkpoints)}")
    print(f"Total files downloaded: {total_files}")
    print(f"Output directory: checkpoints/")
    print(f"\nTip: The last checkpoint is usually the final one. Use it for deployment unless you have a reason to pick an earlier one.")


if __name__ == "__main__":
    main()
