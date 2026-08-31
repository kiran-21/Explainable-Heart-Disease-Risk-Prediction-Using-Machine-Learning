from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from heart_disease_template.config import MODELS_DIR, PROCESSED_DATA_DIR,INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features_heart.csv",
    # labels_path: Path = RAW_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
    df=pd.read_csv(features_path)
    
    # print(df.head(5))
    print("model Training.......")
    
if __name__ == "__main__":
    app()
