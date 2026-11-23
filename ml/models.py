from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

class LastValueRegressor:
    """Predict te next month price, using the current month price"""

    def fit(self,X,y):
        return self

    def predict(self, X):
        return X["valor"].to_numpy()


def get_models():
    return {
        "baseline": LastValueRegressor(),

        "catboost": CatBoostRegressor(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            loss_function="MAE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        ),
        
        "lightgbm": LGBMRegressor(
            n_estimators=200,
            num_leaves=15,
            max_depth=-1,
            learning_rate=0.05,
            objective="regression_l1",
            random_state=42,
            verbosity=-1,
        ),
    }