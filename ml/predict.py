import joblib
from pathlib import Path

from ml.features import FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

def load_model(horizon: int):
  model_path = MODEL_DIR / f"fipe_model_{horizon}m.joblib"

  if not model_path.exists():
    raise FileNotFoundError(
      f"Trained model for {horizon}-month horizon not found: {model_path}"
    )
  
  model = joblib.load(model_path)
  
  return model
