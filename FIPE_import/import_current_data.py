import pandas as pd
from ..db.postgres import get_db_engine
from datetime import datetime


def import_fipe_data():
    ano = datetime.now().year
    mes = datetime.now().month
    mes_str = f"{mes:02d}"

    url_csv = f"https://huggingface.co/datasets/alanwgt/fipex-veiculos-brasil/resolve/main/{ano}/{mes_str}/fipex-prices.csv"

    df = pd.read_csv(url_csv)

    df['valor'] = df['valor_centavos'] / 100
    df.where(pd.notnull(df),None)

    df.to_sql(
        name='fipe',
        con=get_db_engine(),
        if_exists='append',
        index=False
    )