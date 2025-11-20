from train.train_model import train_stat_model

def main():
    for stat in ["points", "rebounds", "assists"]:
        train_stat_model(stat)

    print("🎉 ALL MODELS TRAINED")


if __name__ == "__main__":
    main()