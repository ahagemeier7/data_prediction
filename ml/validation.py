from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from ml.dataset import prepare_dataset
from ml.models import get_models
from ml.features import CATEGORICAL_FEATURES, FEATURES

# ----------------------------------
# Prepare the dataset
# ----------------------------------

df = pd.read_parquet("./FIPE_import/fipex-prices-latest-merged.parquet")

# Keep the original dataset column names for compatibility.
df["valor"] = df["valor_centavos"] / 100

df = prepare_dataset(df)

X = df[FEATURES].copy()
# Convert categorical features to pandas categories for the models.
X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].astype("category")

y = df["target"]

# ----------------------------------
# Select the evaluation months
# ----------------------------------

# Collect distinct target months and sort them chronologically.
months = pd.DatetimeIndex(df["proximo_periodo"].unique()).sort_values()

# Require enough target months for training, three splits, and a final test.
if len(months) < 3 + 2:
    raise ValueError("At least 5 distinct target months are required.")

# Reserve the most recent target month for the final test.
test_month = months[-1]

# Select the three months immediately before the test month.
split_months = months[-(3 + 1):-1]

# Evaluate only the first split during quick experiments.
evaluation_months = split_months[:1]

models = get_models()
results = []

# ----------------------------------
# Train and evaluate the models
# ----------------------------------

#Creating a separate training dataset from earlier months
for evaluation_month in evaluation_months:

    # Train only on targets from earlier months.
    train_mask = df["proximo_periodo"] < evaluation_month

    # Evaluate all targets belonging to the selected month.
    evaluation_mask = df["proximo_periodo"] == evaluation_month

    #filtering the data acording to the mask 
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_evaluation = X.loc[evaluation_mask]
    y_evaluation = y.loc[evaluation_mask]

    print(
        f"\nEvaluation month: {evaluation_month:%Y-%m} | "
        f"Training: {len(X_train):,} rows | "
        f"Evaluation: {len(X_evaluation):,} rows"
    )

    for name, base_model in models.items():

        # Create a fresh model with the same parameters.
        # Each split starts training from scratch.
        model = clone(base_model)

        #Measuring execution time
        start = perf_counter()
        
        #Training the models
        if name == "catboost":
            model.fit(
                X_train,
                y_train,
                cat_features=CATEGORICAL_FEATURES,
            )
        else:
            model.fit(X_train, y_train)

        predictions = model.predict(X_evaluation)

        # MAE is the mean absolute error.
        # Both the target and the error are measured in Brazilian reais.
        mae = mean_absolute_error(y_evaluation, predictions)
        seconds = perf_counter() - start

        results.append({
            "model": name,
            "month": evaluation_month,
            "mae": mae,
            "seconds": seconds,
        })

        print(
            f"{name}: MAE=R$ {mae:,.2f} | "
            f"Time={seconds:.1f}s"
        )

# ----------------------------------
# Summarize the evaluated splits
# ----------------------------------

results_df = pd.DataFrame(results)

summary = results_df.groupby("model").agg(
    mean_mae=("mae", "mean"),
    worst_mae=("mae", "max"),
    total_time_seconds=("seconds", "sum"),
)

print(summary.round(2))
