from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from loguru import logger
from tqdm import tqdm
import typer

from heart_disease_template.config import FIGURES_DIR, PROCESSED_DATA_DIR, INTERIM_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "features_heart.csv",
    interim_path: Path = INTERIM_DATA_DIR/"interim_data_heart.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
   print("Plot and figures..")
   df=pd.read_csv(input_path)
   df2=pd.read_csv(interim_path)
   correlation_matrix(df)
   heart_disease(df2)


def correlation_matrix(df):
    print("checking correlation of data....")
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(14, 14))
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Heatmap", fontsize=16,color="darkblue",fontweight="bold")
    plt.savefig(FIGURES_DIR / "Feature_Correlation_Heatmap.png")
    plt.close()

def heart_disease(df2):
    print("checking the heart disease count based on gender")
    plt.figure(figsize=(12, 8))
    sns.countplot(data=df2, x='HeartDisease', hue='Sex', palette='Set1')
    plt.title("Heart Disease Count by Sex")
    plt.savefig(FIGURES_DIR / "Heart_Disease_Count_by_Sex.png")
    plt.close()


if __name__ == "__main__":
    app()
