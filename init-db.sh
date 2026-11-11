
#!/bin/bash
set -euo pipefail

DB_NAME="data_prediction"

echo "Verificando se o banco '$DB_NAME' existe..."

EXISTS=$(psql \
  -v ON_ERROR_STOP=1 \
  -d postgres \
  -tAc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'")

if [ "$EXISTS" = "1" ]; then
    echo "Banco '$DB_NAME' já existe."
else
    echo "Banco não encontrado. Criando '$DB_NAME'..."

    createdb --maintenance-db=postgres "$DB_NAME"

    echo "Banco '$DB_NAME' criado com sucesso!"
fi
