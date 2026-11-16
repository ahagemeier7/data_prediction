from sklearn.model_selection import train_test_split

from dataset import prepare_dataset
from models import get_models
from features import FEATURES

df = None

df = prepare_dataset(df)

X = df[FEATURES]
y = df["target"]
