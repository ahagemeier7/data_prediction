import pandas as pd
from ..db.postgres import get_db_engine

def initial_payload():
    df = pd.read_parquet('fipex-prices-latest-merged.parquet')

    df['valor'] = df['valor_centavos'] / 100
    df = df.where(pd.notnull(df), None)

    df.to_sql(
        name='fipe',
        con=get_db_engine(),
        if_exists='replace',
        index=False,
    )