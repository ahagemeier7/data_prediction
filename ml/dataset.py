import pandas as pd

VEHICLE_KEYS = [
  "tipo_veiculo",
  "codigo_fipe",
  "ano_modelo",
  "sigla_combustivel",
  "zero_km"
]

def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
  df = df.copy()
  
  #Create a date column to simplify the time validation
  df["periodo"] = pd.to_datetime({
    "year": df["ano_referencia"],
    "month": df["mes_referencia"],
    "day": 1
  })
  
  # Drop null rows on the input subset
  df = df.dropna(
    subset = VEHICLE_KEYS + ["periodo","valor"]
  )
  
  # Delete duplicates checking the set created by the unique vehicle keys and the date column
  if df.duplicated(VEHICLE_KEYS + ["periodo"]).any():
    df.drop_duplicates()
    #raise ValueError(
    #  "There are duplicated registries by vehicle/month"
    #)
    
  # Order by the informed column
  df = df.sort_values(VEHICLE_KEYS + ["periodo"])
  
  #Group by vehicle keys, and preventing it to resort the keys (we have already sorted in the previous step)
  grouped = df.groupby(VEHICLE_KEYS, sort=False)
  
  # Defining the target. .shift(-1) is used to move the rows one row up, so the target value of the row, represent the next month vehicle value
  df["target"] = grouped["valor"].shift(-1)
  
  #Creating the next period value
  df["proximo_periodo"] = grouped["periodo"].shift(-1)
  
  # Calculating the next month  of every registry
  next_month = df["periodo"] + pd.DateOffset(months=1)
  
  # Keeping only the consecutive months
  df = df[
    df["proximo_periodo"].eq(next_month)
  ].copy()
  
  df["idade_veiculo"] = (
    df["ano_referencia"] - df["ano_modelo"]
  )
  
  return df