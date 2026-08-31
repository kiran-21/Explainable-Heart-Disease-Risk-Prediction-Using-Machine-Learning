from pathlib import Path
from loguru import logger
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from heart_disease_template.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

import typer
import pandas as pd
import numpy as np

app = typer.Typer()


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "interim_data_heart.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features_heart.csv",
):
    
   df=pd.read_csv(input_path)
   df = encoding(df)
   df.to_csv(output_path, index=False)
   print("Preprocessed dataset created...check in data folder under processed folder...")
#    print(df)

def encoding(df):
    cat_col=['Sex','ChestPainType','RestingECG','ExerciseAngina', 'ST_Slope']
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[cat_col])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_col))
    df = df.drop(columns=cat_col)
    df = pd.concat([df, encoded_df], axis=1)
    return df

if __name__ == "__main__":
    app()




