from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from heart_disease_template.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "heart.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    print("this is dataset file is loading .....")
    df = pd.read_csv(RAW_DATA_DIR / "heart.csv")
    print(df.head(5))
    print("this is the dataset..")


if __name__ == "__main__":
    app()
