import pandas as pd
from joblib import dump
from pathlib import Path
from sklearn.metrics import mean_absolute_error

from db.postgres import get_db_engine
from ml.models import get_models
from ml.features import prepare_features
from ml.dataset import prepare_dataset

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
  
  months = sorted(df["proximo_periodo"].unique())
  
  if len(months) < 2:
    raise ValueError(
      "Not enough months for training and testing"
    )
    
  test_month = months[-1]
  
  train_df = df[
    df["proximo_periodo"] < test_month
  ].copy()
  
  test_df = df[df["proximo_periodo"] == test_month].copy()
  
  X_train = prepare_features(train_df,SELECTED_MODEL)
  
  y_train = train_df["target"]
  
  X_test = prepare_features(test_df,SELECTED_MODEL)
  
  y_test = test_df["target"]
  
  model = get_models()[SELECTED_MODEL]
  
  model = model.fit(X_train,y_train)
  
  predictions = model.predict(X_test)
  
  mae = mean_absolute_error(
    y_test,predictions
  )
  
  print(f"Model: {SELECTED_MODEL}")
  print(f"Test month: {pd.Timestamp(test_month):%Y-%m}")
  print(f"Test MAE: R$ {mae:,.2f}")
  
  
  X_all = prepare_features(df,SELECTED_MODEL)
  
  y_all = df["target"]
  
  final_model = get_models()[SELECTED_MODEL]
  
  final_model = final_model.fit(X_all,y_all)
  
  MODEL_DIR.mkdir(parents=True, exist_ok=True)

  model_path = MODEL_DIR / "fipe_model.joblib"
  
  dump(final_model,model_path)
  
  
if __name__ == "__main__":
  train()
  