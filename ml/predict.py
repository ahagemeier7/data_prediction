
from pathlib import Path
from functools import lru_cache

import joblib
import pandas as pd

from dataset import HORIZONS
from features import prepare_model_features


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_NAME = "catboost"

@lru_cache(maxsize=4)
def load_model(horizon: int):
  model_path = MODEL_DIR / f"fipe_model_{horizon}m.joblib"

  if not model_path.exists():
    raise FileNotFoundError(
      f"Trained model for {horizon}-month horizon "
      f"not found: {model_path}"
    )

  return joblib.load(model_path)

def predict_vehicle_prices(vehicle: pd.DataFrame) -> dict:
  """
  Predicts vehicle prices for 1, 3, 6 and 12 months.
  """

  # We expect only one vehicle record
  if len(vehicle) != 1:
    raise ValueError(
      "Expected exactly one vehicle record."
    )

  vehicle = vehicle.copy()

  # Convert cents to reais if necessary
  if "valor" not in vehicle.columns:
    vehicle["valor"] = (
      vehicle["valor_centavos"] / 100
    )

  valor_atual = float(vehicle["valor"].iloc[0])

  if not 0 < valor_atual < float("inf"):
    raise ValueError("Invalid current vehicle price.")

  # Get the reference date
  ano = int(vehicle["ano_referencia"].iloc[0])
  mes = int(vehicle["mes_referencia"].iloc[0])

  periodo_atual = pd.Timestamp(
    year=ano,
    month=mes,
    day=1
  )

  # Prepare model features
  X = prepare_model_features(
    vehicle,
    MODEL_NAME
  )

  # Load the trained models
  models = load_model()

  predictions = {}

  # Predict each forecasting horizon
  for horizon in HORIZONS:

    model = models[horizon]

    # Generate prediction
    valor_previsto = float(
        model.predict(X)[0]
    )

    # Calculate the predicted month
    periodo_previsto = (
      periodo_atual
      + pd.DateOffset(months=horizon)
    )

    # Calculate expected price variation
    variacao_percentual = (
      (valor_previsto - valor_atual)
      / valor_atual
    ) * 100

    predictions[f"{horizon}m"] = {
        "mes_previsto": (
          periodo_previsto.strftime("%Y-%m")
        ),
        "valor_previsto": round(
          valor_previsto, 2
        ),
        "variacao_percentual": round(
          variacao_percentual, 2
        ),
    }

  return {
    "codigo_fipe": str(
      vehicle["codigo_fipe"].iloc[0]
    ),
    "mes_atual": periodo_atual.strftime("%Y-%m"),
    "valor_atual": valor_atual,
    "predictions": predictions
  }