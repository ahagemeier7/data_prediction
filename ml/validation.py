from sklearn.model_selection import cross_val_score
import pandas as pd

from dataset import prepare_dataset
from models import get_models
from features import CATEGORICAL_FEATURES, FEATURES

#=====================#
#       Test          #
#=====================#
df = pd.read_parquet("./FIPE_import/fipex-prices-latest-merged.parquet")
df['valor'] = df['valor_centavos'] / 100

df = prepare_dataset(df)

X = df[FEATURES].copy()
#converts the categorical features into pandas category, so the models can use them
X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype("category")
y = df["target"]

models = get_models()

scores = []

for name,model in models.items():
    params = {"cat_features": CATEGORICAL_FEATURES} if name == "catboost" else None
    score = cross_val_score(model, X, y, cv=5, params=params)

    scores.append((name,score.mean()))


print(scores)