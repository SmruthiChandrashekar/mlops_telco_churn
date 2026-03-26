import pandas as pd
from preprocess import preprocess_data


def main():
    # Load raw data
    df = pd.read_csv("raw/dataset.csv")

    # Preprocess
    df_clean = preprocess_data(df)

    # Save processed data
    df_clean.to_csv("processed/cleaned_data.csv", index=False)


if __name__ == "__main__":
    main()