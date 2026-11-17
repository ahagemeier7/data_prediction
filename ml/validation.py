from sklearn.model_selection import cross_val_score
import pandas as pd

from dataset import prepare_dataset
from models import get_models
from features import FEATURES


df = pd.read_parquet("./FIPE_import/fipex-prices-latest-merged.parquet")

df['valor'] = df['valor_centavos'] / 100

df = prepare_dataset(df)

X = df[FEATURES]
y = df["target"]

models = get_models()

scores = []

for name,model in models.items():
    score = cross_val_score(model,X,y,cv=5)

    scores.append((name,score.mean()))


print(scores)