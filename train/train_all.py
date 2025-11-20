# train/train_all.py

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