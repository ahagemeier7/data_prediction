import joblib
from pathlib import Path

from ml.features import FEATURES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "fipe_model.joblib"

def load_model():

  if not MODEL_PATH.exists():
    raise FileNotFoundError(
      "Trained model not found"
    )
  
  model = joblib.load(MODEL_PATH)
  
  return model
  