import pandas as pd
from joblib import dump
from pathlib import Path

from db.postgres import get_db_engine
from ml.models import get_models
from ml.features import prepare_features
from ml.dataset import prepare_dataset,HORIZONS

SELECTED_MODEL = "lightgbm"

DATA_PATH = "./FIPE_import/fipex-prices-latest-merged.parquet"

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

DEV = True

def load_data(dev=DEV) -> pd.DataFrame:
  
  if dev:
    df = pd.read_parquet(DATA_PATH)

    # Convert cents to Brazilian reais
    df["valor"] = df["valor_centavos"] / 100

    # Creates features and next-month target
    df = prepare_dataset(df)
    
  else:
    df = pd.read_sql(
      sql="",
      con=get_db_engine()
    )
  
  return df

def train():
  df = load_data()
  
  MODEL_DIR.mkdir(parents=True, exist_ok=True)

  for horizon in HORIZONS:
    target = f'target_{horizon}m'

    train_df = df.dropna(subset=[target])

    if train_df.empty:
      raise ValueError(
        f"No training data avaliable for horizon {horizon}"
      )

    X = prepare_features(
      train_df,SELECTED_MODEL
    )

    y = train_df[target]
  
    model = get_models()[SELECTED_MODEL]

    model.fit(X,y)

    model_path = MODEL_DIR / f"fipe_model_{horizon}m.joblib"

    dump(model,model_path)


if __name__ == "__main__":
  train()