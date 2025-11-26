#!/usr/bin/env python3
"""
Multi-Model Sycophancy Pipeline Runner
======================================
Orchestrates the full pipeline for one or more models.

Usage:
    python run_pipeline.py                           # Run all models
    python run_pipeline.py --model qwen3_0.6b       # Run single model
    python run_pipeline.py --model qwen3_4b --resume  # Resume (skip completed steps)
    python run_pipeline.py --full-dataset           # Use 4000 questions

Steps:
    1. sycophancy_probe.py      - Generate responses, extract hidden states
    2. judge_sycophancy.py      - LLM judge for sycophancy labels
    3. train_sycophancy_probe.py - Train linear probes
    4. intervention_test.py     - Test steering intervention
    5. generate_plots.py        - Create visualizations
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from model_utils import load_config, get_output_dir

# Pipeline steps in order
PIPELINE_STEPS = [
    {
        "script": "sycophancy_probe.py",
        "log_name": "01_generate_responses.log",
        "output_files": ["sycophancy_results.csv", "sycophancy_hidden_states.npz"],
    },
    {
        "script": "judge_sycophancy.py",
        "log_name": "02_judge_responses.log",
        "output_files": ["sycophancy_judged.csv"],
    },
    {
        "script": "train_sycophancy_probe.py",
        "log_name": "03_train_probes.log",
        "output_files": ["sycophancy_probes.pkl"],
    },
    {
        "script": "intervention_test.py",
        "log_name": "04_intervention.log",
        "output_files": ["intervention_results.csv"],
    },
    {
        "script": "generate_plots.py",
        "log_name": "05_generate_plots.log",
        "output_files": ["plots"],  # directory
    },
]


def get_log_dir(model_key: str) -> Path:
    """Get log directory for a model."""
    log_dir = Path(__file__).parent / "logs" / model_key
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def step_completed(model_key: str, step: dict) -> bool:
    """Check if a step has completed by checking for output files."""
    output_dir = get_output_dir(model_key)
    for output_file in step["output_files"]:
        path = output_dir / output_file
        if not path.exists():
            return False
    return True


def run_step(model_key: str, step: dict, full_dataset: bool) -> bool:
    """
    Run a single pipeline step.
    Tees stdout/stderr to both terminal and log file with timestamps.
    Returns True if successful.
    """
    script = step["script"]
    log_name = step["log_name"]
    log_dir = get_log_dir(model_key)
    log_path = log_dir / log_name

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running: {script}")
    print(f"Model: {model_key}")
    print(f"Log: {log_path}")
    print(f"{'='*60}\n")

    # Set environment variables
    env = os.environ.copy()
    env["MODEL_KEY"] = model_key
    if full_dataset:
        env["USE_FULL_DATASET"] = "1"

    # Open log file
    with open(log_path, "w") as log_file:
        # Write header
        log_file.write(f"{'='*60}\n")
        log_file.write(f"Script: {script}\n")
        log_file.write(f"Model: {model_key}\n")
        log_file.write(f"Started: {datetime.now().isoformat()}\n")
        log_file.write(f"Full dataset: {full_dataset}\n")
        log_file.write(f"{'='*60}\n\n")
        log_file.flush()

        # Run script with real-time output
        script_path = Path(__file__).parent / script
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Tee output to terminal and log file
        for line in process.stdout:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            timestamped_line = f"[{timestamp}] {line}"

            # Print to terminal (original line, no timestamp for cleaner output)
            print(line, end='')
            sys.stdout.flush()

            # Write timestamped line to log
            log_file.write(timestamped_line)
            log_file.flush()

        process.wait()

        # Write footer
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Finished: {datetime.now().isoformat()}\n")
        log_file.write(f"Exit code: {process.returncode}\n")
        log_file.write(f"{'='*60}\n")

    if process.returncode != 0:
        print(f"\n[ERROR] {script} failed with exit code {process.returncode}")
        return False

    return True


def run_pipeline(model_key: str, resume: bool = False, full_dataset: bool = False):
    """Run the full pipeline for a single model."""
    config = load_config()

    if model_key not in config["models"]:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available models: {list(config['models'].keys())}")
        sys.exit(1)

    model_config = config["models"][model_key]
    print(f"\n{'#'*60}")
    print(f"# Pipeline: {model_key}")
    print(f"# HuggingFace: {model_config['hf_name']}")
    print(f"# Thinking mode: {model_config['thinking_mode']}")
    print(f"# Full dataset: {full_dataset}")
    print(f"# Resume: {resume}")
    print(f"{'#'*60}")

    for step in PIPELINE_STEPS:
        if resume and step_completed(model_key, step):
            print(f"\n[SKIP] {step['script']} - outputs already exist")
            continue

        success = run_step(model_key, step, full_dataset)
        if not success:
            print(f"\n[ABORT] Pipeline stopped due to error in {step['script']}")
            sys.exit(1)

    print(f"\n{'#'*60}")
    print(f"# Pipeline completed: {model_key}")
    print(f"# Results: results/{model_key}/")
    print(f"# Logs: logs/{model_key}/")
    print(f"{'#'*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run sycophancy probing pipeline for multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                           # Run all models
    python run_pipeline.py --model qwen3_0.6b       # Run single model
    python run_pipeline.py --model qwen3_4b --resume  # Resume incomplete run
    python run_pipeline.py --list                   # List available models
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Run pipeline for specific model only"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip steps whose output files already exist"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Use full 4000-question dataset instead of 1000"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and exit"
    )

    args = parser.parse_args()
    config = load_config()

    if args.list:
        print("Available models:")
        for key, cfg in config["models"].items():
            print(f"  {key}: {cfg['hf_name']} (thinking: {cfg['thinking_mode']})")
        sys.exit(0)

    # Determine dataset setting
    full_dataset = args.full_dataset or config.get("dataset", {}).get("use_full", False)

    # Run for specified model or all models
    if args.model:
        run_pipeline(args.model, args.resume, full_dataset)
    else:
        for model_key in config["models"].keys():
            run_pipeline(model_key, args.resume, full_dataset)


if __name__ == "__main__":
    main()
