# pipeline/pipeline.py

import sys
import os
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON = sys.executable  # ensures Streamlit Cloud uses correct venv


def run_step(name: str, script_path: str):
    """Runs a pipeline step and streams output."""
    script_path = os.path.join(ROOT_DIR, script_path)

    print(f"=== Running: {name} ===")

    process = subprocess.Popen(
        [PYTHON, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    out, err = process.communicate()

    if out:
        print("OUTPUT:\n", out)
    if err:
        print("ERRORS:\n", err)


def run_full_pipeline():
    run_step("Scraping Logs", "scrapers/scrape_basic_logs.py")
    run_step("Building Dataset", "scrapers/build_dataset.py")
    run_step("Training Models", "train/train_all.py")
