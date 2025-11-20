from train.train_model import train_stat_model

def main():
    print("Training POINTS...")
    train_stat_model("points")
    print("Training REBOUNDS...")
    train_stat_model("rebounds")
    print("Training ASSISTS...")
    train_stat_model("assists")

if __name__ == "__main__":
    main()