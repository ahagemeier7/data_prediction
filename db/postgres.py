import os
import sys
import logging
from sqlalchemy import create_engine, Engine

_engine = None

def get_db_engine():
    global _db_engine

    if _db_engine:
        return _db_engine
    
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    db_server = os.getenv("DB_SERVER")

    try:
        _db_engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_server}/{db_name}")
        return _db_engine
    except Exception as e:
        logging.error(f"Error creating database engine: {e}")
        sys.exit