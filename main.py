from FIPE_import.initial_payload import initial_payload
from FIPE_import.import_current_data import import_fipe_data
from db.postgres import get_db_engine
from sqlalchemy import text
from datetime import datetime

def initial_payload_check() -> int:
    with get_db_engine().connect() as conn:
        total_rows: int = conn.scalar(text("select count(*) from fipe"))

    return total_rows

def get_last_import() -> tuple[int,int]:

    engine = get_db_engine()

    query = text("""
        SELECT ano_referencia, mes_referencia
        FROM fipe
        WHERE ano_referencia IS NOT NULL
          AND mes_referencia IS NOT NULL
        ORDER BY ano_referencia DESC, mes_referencia DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query).first()

    if result is None:
        return None

    return int(result[0]), int(result[1])

def import_mising_fipe_data():
    last_import = get_last_import()

    ano,mes = last_import
    
    current_year = datetime.now().year
    current_month = datetime.now().month

    if (ano,mes) == (current_year,current_month):
        print("All data is imported")
        return

    if mes == 12:
        ano += 1
        mes = 1
    else:
        mes += 1

    while (ano,mes) <= (current_year,current_month):
        try:
            import_fipe_data(ano,mes)

        except Exception as e:
            print(f"Import failed on date {mes:02d}/{ano}: {e}")
            raise

        if mes == 12:
            ano += 1
            mes = 1
        else:
            mes += 1

    print("All missing data was imported")


if __name__ == "__main__":

    try:
        if initial_payload_check() <= 0:
            initial_payload()

        import_mising_fipe_data()

    except Exception as e:
        pass
