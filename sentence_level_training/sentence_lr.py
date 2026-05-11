
import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, Normalizer

from sklearn.metrics import mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr

from tqdm import tqdm


def fix_random(RANDOM_SEED):
    np.random.seed(RANDOM_SEED)

# Normalization functions
def normalize(x, norm, min_value, max_value):
    if norm == 'min_max':
        print("Applying min max scaler on pseudo scores")
        scores = MinMaxScaler((float(min_value), float(max_value))).fit_transform(x).ravel()
        return scores

    elif norm == 'znorm':
        print("Applying z norm + min max scaler on pseudo scores")
        z = StandardScaler().fit_transform(x)
        scores = MinMaxScaler((0.0, float(max_value))).fit_transform(z).ravel()
        return scores

    elif norm == 'tanh':
        print("Applying z norm + tanh + min max scaler on pseudo scores")
        z = StandardScaler().fit_transform(x)
        t = np.tanh(z)
        scores = MinMaxScaler((0.0, float(max_value))).fit_transform(t).ravel()
        return scores

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Drops zero-std features (computed on fit data) and removes highly correlated
    features (above corr_threshold), storing exactly which columns were kept.
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

        # Drop zero-std features
        stds = X.std(numeric_only=True)
        non_zero_std_cols = stds[stds > 0].index.tolist()
        self.zero_std_cols_ = [c for c in self.input_feature_names_ if c not in non_zero_std_cols]
        X_nz = X[non_zero_std_cols]

        # Remove correlated features
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

        # align columns by name; missing columns (if any) will raise an error
        missing = [c for c in self.keep_cols_ if c not in X.columns]
        if missing:
            raise ValueError(f"Transform input is missing columns seen during fit: {missing[:5]} {'...' if len(missing)>5 else ''}")

        return X[self.keep_cols_]

# MAIN FUNCTION
def train_sentence_lr(df, df_sets, config):
    # ==== HYPERPARAMETERS ====
    RANDOM_SEED = config.random_seed

    PROMPT = config.prompt
    TARGET_COL = config.target_col
    ESSAY_COL = config.essay_col
    INDEX_COL = config.index_col
    MODEL_NAME = config.model_name
    ALPHA = config.alpha
    ADD_BIAS = config.add_bias
    APPLY_NORM = config.apply_norm
    ABLATION = config.ablation
    BASELINE = config.baseline
    INPUT_FILENAME = config.input_filename

    window_size = config.window_size
    sentence_file = os.path.join('data', 'processed', INPUT_FILENAME[:4]+f'_set{PROMPT}_loosingle_with_features.csv')

    corr_threshold = 0.9

    if config.window_size == 1:
        loo_predictions_path = os.path.join('loo_predictions', MODEL_NAME+'_loosingle.csv')
    else:
        loo_predictions_path = os.path.join('loo_predictions', MODEL_NAME+f'_loomulti_{window_size}.csv')

    min_value = df[TARGET_COL].min()
    max_value = df[TARGET_COL].max()

    print("="*30+"\n")
    print("TRAINING SENTENCE LEVEL LINEAR REGRESSOR")
    print("Correlation threshold", corr_threshold)

    print("Loading loo predictions:", loo_predictions_path)
    dfps = pd.read_csv(loo_predictions_path)

    # fix random
    fix_random(RANDOM_SEED)

    if not BASELINE:
        print("\nComputing pseudo scores using loo predictions!")

        if ADD_BIAS:
            bias = (dfps['pred_loo'] - dfps['pred_essay']).mean()
            print("bias:", bias)
            dfps['pseudo_scores'] = dfps[TARGET_COL] - ALPHA * (dfps['pred_loo'] - dfps['pred_essay'] - bias)
        else:
            dfps['pseudo_scores'] = dfps[TARGET_COL] - ALPHA * (dfps['pred_loo'] - dfps['pred_essay'])

        print("Distribution of scores:")
        print(dfps['pseudo_scores'].round().value_counts().sort_index())
        print(dfps['pseudo_scores'].round().value_counts(normalize=True).sort_index())

        if APPLY_NORM in ["min_max", "znorm", "tanh"]:
            dfps['pseudo_scores'] = normalize(dfps['pseudo_scores'].to_numpy().reshape(-1,1), APPLY_NORM, min_value, max_value)
            print("Distribution of scores after normalization:")
            print(dfps['pseudo_scores'].round().value_counts().sort_index())
            print(dfps['pseudo_scores'].round().value_counts(normalize=True).sort_index())
        else:
            print("Normalization not applied!")
    else:
        print("Running BASELINE!")
        print("Propagating essay score to each sentence")
        dfps['pseudo_scores'] = dfps[TARGET_COL]

    sentence_model_name = f"{MODEL_NAME}_ablation{ABLATION}_alpha{int(ALPHA)}_norm{APPLY_NORM}"
    if ADD_BIAS:
        sentence_model_name += "_withbias_sentence"
    else:
        sentence_model_name += "_withoutbias_sentence"

    if BASELINE:
        sentence_model_name += "_baseline"

    print("Model name", sentence_model_name)

    # save trained model to model_path
    sentence_model_path = os.path.join('models', sentence_model_name+'.pkl')

    # save predictions using sentence-level trained model to predictions_path
    sentence_predictions_path = os.path.join('sentence_predictions', sentence_model_name+'.csv')

    # min and max pseudo scores
    min_possible_value = dfps['pseudo_scores'].min()
    max_possible_value = dfps['pseudo_scores'].max()
    print("Min and max values:", min_possible_value, max_possible_value)

    # indices of splits ESSAY LEVEL
    train_index = df_sets['train_df'][INDEX_COL].values.tolist()
    val_index = df_sets['val_df'][INDEX_COL].values.tolist()
    test_index = df_sets['test_df'][INDEX_COL].values.tolist()
    trainval_index = train_index + val_index
    print(f"Essay level --> Train: {len(trainval_index)} | Test: {len(test_index)}")

    # loading linguist features at sentence level
    lingfeats_df = pd.read_csv(sentence_file)

    # feature slice: from "count_words" onward, expect 220 features
    if "count_words" not in lingfeats_df.columns:
        raise KeyError('"count_words" column not found in DataFrame columns.')

    start_idx = lingfeats_df.columns.get_loc("count_words")
    lingfeat_cols = list(lingfeats_df.columns[start_idx:start_idx + 220])

    if len(lingfeat_cols) != 220:
        raise ValueError(f"Linguistic features dimension is not 220, but {len(lingfeat_cols)}")

    # filter and keep lingfeats and merging columns
    keep_cols = lingfeat_cols + [INDEX_COL, 'sentence_index']
    lingfeats_df = lingfeats_df[keep_cols]

    # merging lingfeats and target sentence scores
    dfps = dfps.merge(lingfeats_df, on=[INDEX_COL, "sentence_index"], how="inner")

    # drop nan rows
    before = len(dfps)
    dfps = dfps.dropna(subset=lingfeat_cols)
    dropped = before - len(dfps)
    print(f"Dropped {dropped} rows due to missing values.")

    # filtering splits by ids SENTENCE LEVEL
    trainval_df = dfps[dfps[INDEX_COL].isin(trainval_index)]
    test_df = dfps[dfps[INDEX_COL].isin(test_index)]
    print(f"Sentence level --> Train: {len(trainval_df)} | Test: {len(test_df)}")

    X_train = trainval_df[lingfeat_cols].copy()
    y_train = trainval_df['pseudo_scores'].copy()
    X_test = test_df[lingfeat_cols].copy()
    y_test = test_df['pseudo_scores'].copy()

    # pipeline (feature selection + scaling + model)
    pipe = Pipeline(steps=[
        ("select", FeatureSelector(corr_threshold=corr_threshold)),
        ("scale", MinMaxScaler()),
        ("model", LinearRegression())
    ])

    # train
    pipe.fit(X_train, y_train)

    # predict
    preds = pipe.predict(X_test)

    # save pipeline (preprocessing + model)
    joblib.dump(pipe, sentence_model_path)
    print("Model saved to:", sentence_model_path)

    # save predictions
    essay_preds = pd.DataFrame({
        'index': test_df.index,
        INDEX_COL: test_df[INDEX_COL],
        'sentence_index': test_df['sentence_index'],
        'prompt': PROMPT,
        'target': TARGET_COL,
        'prediction': preds,
        'pseudo_scores': y_test,
    })
    essay_preds.to_csv(sentence_predictions_path, index=False)
    print("Predictions saved to:", sentence_predictions_path)

    # METRICS
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
        f"\nFinal Test Sentence-level"
        f"RMSE: {rmse:.4f} | "
        f"R2 Score: {r2:.4f} | "
        f"Pearson r: {pearson_r:.4f} (p={pearson_p:.4f}) | "
        f"QWK Rounded: {qwk:.4f} | "
        f"QWK Rescaled: {qwk_rescaled:.4f}"
    )

    return results


if __name__ == "__main__":
    pass
