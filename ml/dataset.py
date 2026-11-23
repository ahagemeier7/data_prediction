import pandas as pd

VEHICLE_KEYS = [
    "tipo_veiculo",
    "codigo_fipe",
    "ano_modelo",
    "sigla_combustivel",
    "zero_km"
]

HORIZONS = [1, 3, 6, 12]

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()

  # Create a date column for time validation
  df["periodo"] = pd.to_datetime({
      "year": df["ano_referencia"],
      "month": df["mes_referencia"],
      "day": 1
  })

  # Drop rows with missing required values
  df = df.dropna(
      subset=VEHICLE_KEYS + ["periodo", "valor"]
  )

  # Remove duplicated vehicle/month records
  df = df.drop_duplicates(
      subset=VEHICLE_KEYS + ["periodo"]
  )

  # Sort records by vehicle and chronological order
  df = df.sort_values(VEHICLE_KEYS + ["periodo"])

  # Group records by vehicle
  grouped = df.groupby(VEHICLE_KEYS, sort=False)

  # Create targets for each forecasting horizon
  for horizon in HORIZONS:

    # Price of the future record
    df[f"target_{horizon}m"] = (
        grouped["valor"].shift(-horizon)
    )

    # Date of the future record
    future_period = (
        grouped["periodo"].shift(-horizon)
    )

    # Expected date for this forecasting horizon
    expected_period = (
        df["periodo"]
        + pd.DateOffset(months=horizon)
    )

    # Remove targets that do not match the requested calendar horizon.
    df[f"target_{horizon}m"] = (
        df[f"target_{horizon}m"].where(
          future_period.eq(expected_period)
        )
    )

  # Calculate vehicle age
  df["idade_veiculo"] = (
    df["ano_referencia"] - df["ano_modelo"]
  )

  return df
