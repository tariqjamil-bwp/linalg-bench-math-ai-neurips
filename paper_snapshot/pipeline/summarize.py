#!/usr/bin/env python3
"""
Summarize accuracy from inference results and append to consolidated accuracy file.

Usage:
    python pipeline/summarize.py <input_csv> <model> <results_jsonl>
"""

import os
import sys
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def update_accuracy_file(input_csv: str, model: str, results_file: str):
    """Append model accuracy to consolidated accuracy_{size}.csv."""
    # Determine size from input filename
    input_name = os.path.basename(input_csv)
    if "3x3" in input_name:
        size = "3x3"
    elif "4x4" in input_name:
        size = "4x4"
    elif "5x5" in input_name:
        size = "5x5"
    else:
        logger.warning(f"  Cannot determine size from {input_name}")
        return

    # Get data directory
    data_dir = os.path.dirname(os.path.abspath(input_csv))
    results_dir = os.path.join(data_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    accuracy_file = os.path.join(results_dir, f"accuracy_{size}.csv")

    # Count Pass/Fail from results
    pass_count = 0
    fail_count = 0
    total = 0

    if not os.path.exists(results_file):
        logger.warning(f"  Results file not found: {results_file}")
        return

    with open(results_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                total += 1
                if record.get("correct"):
                    pass_count += 1
                else:
                    fail_count += 1
            except json.JSONDecodeError:
                continue

    accuracy = (pass_count / total * 100) if total > 0 else 0

    # Create or append to accuracy file
    header = "Model,Pass,Fail,Accuracy%"
    new_row = f"{model},{pass_count},{fail_count},{accuracy:.1f}"

    if not os.path.exists(accuracy_file):
        with open(accuracy_file, "w") as f:
            f.write(header + "\n" + new_row + "\n")
        logger.info(f"  Created: {accuracy_file}")
    else:
        # Check if model already exists
        with open(accuracy_file, "r") as f:
            lines = f.readlines()
        existing_models = [line.split(",")[0] for line in lines[1:] if line.strip()]
        if model in existing_models:
            # Always overwrite unless resuming - rebuild without this model
            with open(accuracy_file, "w") as f:
                f.write(header + "\n")
                for l in lines[1:]:
                    if not l.startswith(model + ","):
                        f.write(l)
            with open(accuracy_file, "a") as f:
                f.write(new_row + "\n")
            logger.info(f"  Overwrote: {accuracy_file}")
        else:
            with open(accuracy_file, "a") as f:
                f.write(new_row + "\n")
            logger.info(f"  Updated: {accuracy_file}")

    logger.info(f"  {model}: {pass_count}/{total} correct ({accuracy:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python summarize.py <input_csv> <model> <results_jsonl>")
        sys.exit(1)

    input_csv = sys.argv[1]
    model = sys.argv[2]
    results_file = sys.argv[3]

    update_accuracy_file(input_csv, model, results_file)