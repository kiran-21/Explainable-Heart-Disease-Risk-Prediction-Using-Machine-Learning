import typer
import pandas as pd
import numpy as np

from pathlib import Path
from loguru import logger
from tqdm import tqdm
from heart_disease_template.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "heart.csv",
    output_path: Path = INTERIM_DATA_DIR / "interim_data_heart.csv",
):
    print("dataset file loading...")
    df = pd.read_csv(input_path)
    print("First five data from the raw dataset")
    print(df.head(5))

    df = cleaning(df)

    df.to_csv(output_path, index=False)
    print("Interim dataset created...check in data folder under Interim folder...")

def cleaning(df):
    df.info()
    #since the dataset have zero null values I am skipping the cleaning dataset
    df = df.copy() #here the dataset is simply copied , no extra cleaning is done so far.
    return df
    

if __name__ == "__main__":
    app()
