# train/train_all.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from train.train_model import train_stat_model


def main():
    print("🔥 Training all models...")

    train_stat_model("points")
    train_stat_model("rebounds")
    train_stat_model("assists")

    print("🎉 All models trained successfully!")