# train/train_all.py

import sys
import os

# ------------------------------------------------------------
# FIX: Ensure this script can import train_model.py from repo root
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from train.train_model import train_stat_model


def main():
    print("Training POINTS...")
    train_stat_model("points")

    print("\nTraining REBOUNDS...")
    train_stat_model("rebounds")

    print("\nTraining ASSISTS...")
    train_stat_model("assists")

    print("\nAll models trained successfully!")


if __name__ == "__main__":
    main()