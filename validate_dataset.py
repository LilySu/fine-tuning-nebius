"""Validate JSONL dataset files for fine-tuning schema correctness."""

import json
import os
import sys

VALID_ROLES = {"system", "user", "assistant"}


def validate_line(line_str, line_num, filename):
    """Validate a single JSONL line. Return list of error strings."""
    errors = []

    try:
        obj = json.loads(line_str)
    except json.JSONDecodeError as e:
        return [f"{filename}:{line_num}: Invalid JSON — {e}"]

    if "messages" not in obj:
        return [f"{filename}:{line_num}: Missing 'messages' key"]

    messages = obj["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        return [f"{filename}:{line_num}: 'messages' must be a list with at least 2 elements"]

    roles_seen = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"{filename}:{line_num}: messages[{i}] is not an object")
            continue
        if "role" not in msg:
            errors.append(f"{filename}:{line_num}: messages[{i}] missing 'role'")
        elif msg["role"] not in VALID_ROLES:
            errors.append(f"{filename}:{line_num}: messages[{i}] invalid role '{msg['role']}'")
        else:
            roles_seen.add(msg["role"])

        if "content" not in msg:
            errors.append(f"{filename}:{line_num}: messages[{i}] missing 'content'")
        elif not isinstance(msg["content"], str) or not msg["content"].strip():
            errors.append(f"{filename}:{line_num}: messages[{i}] 'content' must be a non-empty string")

    if "user" not in roles_seen:
        errors.append(f"{filename}:{line_num}: No message with role 'user'")
    if "assistant" not in roles_seen:
        errors.append(f"{filename}:{line_num}: No message with role 'assistant'")

    return errors


def validate_file(filepath):
    """Validate all lines in a JSONL file. Return (total, valid, invalid, errors)."""
    basename = os.path.basename(filepath)

    try:
        with open(filepath) as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: File not found — {filepath}")
        return 0, 0, 1, [f"{filepath}: File not found"]

    total = 0
    invalid = 0
    all_errors = []

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        errors = validate_line(stripped, line_num, basename)
        if errors:
            invalid += 1
            all_errors.extend(errors)

    valid = total - invalid
    return total, valid, invalid, all_errors


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file1.jsonl> [file2.jsonl ...]")
        sys.exit(1)

    any_errors = False

    for filepath in sys.argv[1:]:
        total, valid, invalid, errors = validate_file(filepath)

        for err in errors:
            print(f"  ERROR: {err}")

        basename = os.path.basename(filepath)
        print(f"\n=== {basename} ===")
        print(f"Total lines: {total}")
        print(f"Valid: {valid}")
        print(f"Invalid: {invalid}")

        if invalid > 0:
            any_errors = True

    sys.exit(1 if any_errors else 0)


if __name__ == "__main__":
    main()
