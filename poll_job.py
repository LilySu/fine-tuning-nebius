"""Poll a Nebius Token Factory fine-tuning job until it reaches a terminal state."""

import json
import os
import sys
import time
from datetime import datetime

from openai import OpenAI

from config import NEBIUS_API_BASE, POLL_INTERVAL, get_api_base

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


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

    print(f"Polling job {job_id} every {POLL_INTERVAL}s...\n")

    try:
        while True:
            job = client.fine_tuning.jobs.retrieve(job_id)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Status: {job.status}", end="")

            if hasattr(job, "trained_steps") and job.trained_steps is not None:
                print(f"  (step {job.trained_steps}/{job.total_steps})", end="")
            print()

            if job.status in TERMINAL_STATUSES:
                break

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nPolling interrupted. Job is still running on Nebius. Re-run this script to resume monitoring.")
        sys.exit(0)

    print()

    if job.status == "succeeded":
        print("Job completed successfully!")
        if hasattr(job, "trained_tokens") and job.trained_tokens is not None:
            print(f"Trained tokens: {job.trained_tokens}")

        try:
            events = client.fine_tuning.jobs.list_events(job_id)
            print("\n--- Job Events ---")
            for event in events.data:
                print(f"  [{event.created_at}] {event.level}: {event.message}")
        except Exception as e:
            print(f"Warning: Could not retrieve job events: {e}")

        sys.exit(0)

    elif job.status == "failed":
        print("Job FAILED.")
        if hasattr(job, "error") and job.error is not None:
            print(f"  Code:    {job.error.code}")
            print(f"  Message: {job.error.message}")
            print(f"  Param:   {job.error.param}")
        print("\nCheck the error above. For transient 5xx errors, re-run create_job.py.")
        sys.exit(1)

    elif job.status == "cancelled":
        print("Job was cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
