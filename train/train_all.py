# train/train_all.py

import sys, os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from train.train_model import train_stat_model


def main():
    print("🔥 Training all models...")

    print("\n➡️ Training POINTS...")
    train_stat_model("points")

    print("\n➡️ Training REBOUNDS...")
    train_stat_model("rebounds")

    print("\n➡️ Training ASSISTS...")
    train_stat_model("assists")

    print("\n🎉 All models trained successfully!")


# DO **NOT** auto-run.
# Streamlit imports this file on startup; auto-running will crash the app.
#
# if __name__ == "__main__":
#     main()