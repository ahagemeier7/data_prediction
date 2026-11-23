import pandas as pd

CATEGORICAL_FEATURES = [
  "codigo_fipe",
  "nome_marca",
  "tipo_veiculo",
  "sigla_combustivel",
]

FEATURES = [
  "codigo_fipe",
  "nome_marca",
  "tipo_veiculo",
  "sigla_combustivel",
  "ano_modelo",
  "idade_veiculo",
  "zero_km",
  "ano_referencia",
  "mes_referencia",
  "valor"
]

def prepare_features(df: pd.DataFrame,model_name:str) -> pd.DataFrame:
  X = df[FEATURES].copy()
  
  for col in CATEGORICAL_FEATURES:

    if model_name == "catboost":
      X[col] = (
        X[col]
        .fillna("unknown")
        .astype(str)
      )

    elif model_name == "lightgbm":
      X[col] = (
        X[col]
        .astype("category")
      )

  return X