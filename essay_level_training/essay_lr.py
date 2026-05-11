import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


from sklearn.metrics import mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr


def fix_random(RANDOM_SEED):
    np.random.seed(RANDOM_SEED)

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy="mean"):
        self.strategy = strategy
        self.imputer_ = SimpleImputer(strategy=strategy)
        self.columns_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameImputer expects a pandas DataFrame.")
        self.columns_ = X.columns
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameImputer expects a pandas DataFrame.")
        arr = self.imputer_.transform(X)
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drops zero-std features (computed on fit data) and removes highly correlated
    features (above corr_threshold), storing which columns were kept.
    """
    def __init__(self, corr_threshold=0.9):
        self.corr_threshold = float(corr_threshold)
        self.keep_cols_ = None
        self.zero_std_cols_ = None
        self.corr_dropped_cols_ = None
        self.input_feature_names_ = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector expects a pandas DataFrame with named columns.")
        self.input_feature_names_ = list(X.columns)

        # drop zero-std features
        stds = X.std(numeric_only=True)
        non_zero_std_cols = stds[stds > 0].index.tolist()
        self.zero_std_cols_ = [c for c in self.input_feature_names_ if c not in non_zero_std_cols]
        X_nz = X[non_zero_std_cols]

        # remove correlated features
        corr = X_nz.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > self.corr_threshold):
                to_drop.add(col)

        self.corr_dropped_cols_ = sorted(list(to_drop))
        self.keep_cols_ = [c for c in non_zero_std_cols if c not in to_drop]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureSelector expects a pandas DataFrame at transform time.")

        # align columns by name; missing columns (if any) will raise error
        missing = [c for c in self.keep_cols_ if c not in X.columns]
        if missing:
            raise ValueError(f"Transform input is missing columns seen during fit: {missing[:5]} {'...' if len(missing)>5 else ''}")

        return X[self.keep_cols_]


def train_essay_lr(df, df_sets, config):
    # ==== HYPERPARAMETERS ====
    RANDOM_SEED = config.random_seed

    PROMPT = config.prompt
    TARGET_COL = config.target_col
    ESSAY_COL = config.essay_col
    INDEX_COL = config.index_col
    MODEL_NAME = config.model_name

    corr_threshold = 0.9

    print("="*30+"\n")
    print("TRAINING ESSAY LEVEL LINEAR REGRESSOR")
    print("Correlation threshold", corr_threshold)

    # fix random
    fix_random(RANDOM_SEED)

    # data
    train_df = df_sets['train_df']
    val_df = df_sets['val_df']
    test_df = df_sets['test_df']

    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Train: {len(trainval_df)} | Test: {len(test_df)}")

    # save trained model to model_path
    model_path = os.path.join('models', MODEL_NAME+'.pkl')

    # save predictions to predictions_path
    predictions_path = os.path.join('essay_predictions', MODEL_NAME+'.csv')

    # min and max values
    min_possible_value = df[TARGET_COL].min()
    max_possible_value = df[TARGET_COL].max()
    print("Min and max values:", min_possible_value, max_possible_value)

    # feature slice: from "count_words" onward, expect 220 features
    if "count_words" not in trainval_df.columns:
        raise KeyError('"count_words" column not found in DataFrame columns.')

    start_idx = trainval_df.columns.get_loc("count_words")
    lingfeat_cols = list(trainval_df.columns[start_idx:start_idx + 220])

    if len(lingfeat_cols) != 220:
        raise ValueError(f"Linguistic features dimension is not 220, but {len(lingfeat_cols)}")

    X_train = trainval_df[lingfeat_cols].copy()
    y_train = trainval_df[TARGET_COL].copy()
    X_test = test_df[lingfeat_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    # build the full pipeline (feature selection + scaling + model)
    pipe = Pipeline(steps=[
        ("impute", DataFrameImputer(strategy="mean")),
        ("select", FeatureSelector(corr_threshold=corr_threshold)),
        ("scale", MinMaxScaler()),
        ("model", LinearRegression())
    ])

    # train
    pipe.fit(X_train, y_train)

    # predict
    preds = pipe.predict(X_test)

    # save pipeline (preprocessing + model)
    joblib.dump(pipe, model_path)
    print("Model saved to:", model_path)

    # save predictions
    essay_preds = pd.DataFrame({
        'index': test_df.index,
        'prompt': PROMPT,
        'target': TARGET_COL,
        'essay_id': test_df[INDEX_COL],
        'prediction': preds,
        'true_score': y_test,
    })
    essay_preds.to_csv(predictions_path, index=False)
    print("Predictions saved to:", predictions_path)

    # metrics
    preds = np.clip(preds, min_possible_value, max_possible_value)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    pearson_r, pearson_p = pearsonr(y_test, preds)

    preds_rounded = np.round(np.clip(preds, min_possible_value, max_possible_value)).astype(int)
    labels_rounded = np.round(y_test).astype(int)
    qwk = cohen_kappa_score(labels_rounded, preds_rounded, weights="quadratic")

    preds_rescaled = np.round(np.clip(preds * 2, min_possible_value * 2, max_possible_value * 2)).astype(int)
    labels_rescaled = np.round(y_test * 2).astype(int)
    qwk_rescaled = cohen_kappa_score(labels_rescaled, preds_rescaled, weights="quadratic")

    results = {
        'rmse': rmse,
        'r2': r2,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'qwk': qwk,
        'qwk_rescaled': qwk_rescaled,
    }

    print(
        f"\nFinal Test "
        f"RMSE: {rmse:.4f} | "
        f"R2 Score: {r2:.4f} | "
        f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4f}) | "
        f"QWK Rounded: {qwk:.4f} | "
        f"QWK Rescaled: {qwk_rescaled:.4f}"
    )

    return results

def predict_with_pipeline(model_path, feats_df, start_col_name="count_words", n_features=220):
    pipe = joblib.load(model_path)

    if start_col_name not in feats_df.columns:
        raise KeyError(f'"{start_col_name}" column not found in DataFrame.')

    start_idx = feats_df.columns.get_loc(start_col_name)
    lingfeat_cols = list(feats_df.columns[start_idx:start_idx + n_features])

    if len(lingfeat_cols) != n_features:
        raise ValueError(f"Expected {n_features} features, got {len(lingfeat_cols)}")

    X = feats_df[lingfeat_cols].copy()
    print("Rows with NaN:", X.isnull().any(axis=1).sum())

    return pipe.predict(X)


if __name__ == "__main__":
    pass
