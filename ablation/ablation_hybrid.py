
import os
import pandas as pd
import numpy as np
import ast
import joblib

import torch

from tqdm import tqdm

from essay_level_training import essay_hybrid


def loo_predictions_exists(filepath):
    if os.path.isfile(filepath):
        print("Loo essay predictions already computed!")
        print("File:", filepath)
        print("="*30+"\n")
        return True
    return False

def predict_essay_loo(sents_df, essays_df, index_loo_col, config):
    INDEX_COL = config.index_col
    ESSAY_COL = config.essay_col
    MODEL_DIR = config.model_dir
    MODEL_NAME = config.model_name

    model_path = os.path.join('models', MODEL_NAME+'.pt')
    scaler_path = os.path.join('models', MODEL_NAME+'_scaler.pkl')

    if config.device == "mps" and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif config.device == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    essay_loo_preds, essay_preds = [], {}

    start_idx = essays_df.columns.get_loc('count_words')
    lingfeat_cols = list(essays_df.columns[start_idx:start_idx + 220])
    assert len(lingfeat_cols) == 220, "linguistic features != 220"

    tokenizer = essay_hybrid.load_tokenizer(MODEL_DIR)
    model = essay_hybrid.load_model(model_path, 220, MODEL_DIR)
    scaler = joblib.load(scaler_path)

    # scaling linguistic features
    essays_df[lingfeat_cols] = scaler.transform(essays_df[lingfeat_cols])
    sents_df[lingfeat_cols] = scaler.transform(sents_df[lingfeat_cols])

    model.to(DEVICE)
    model.eval()

    print(f"Predicting scores on {config.ablation} essay loo")
    with torch.no_grad():
        for _, row in tqdm(sents_df.iterrows(), total=len(sents_df)):
            id = row[INDEX_COL]
            removed_indices = row[index_loo_col]

            if id not in essay_preds.keys():
                # Predict full essay
                text_full = essays_df.loc[essays_df[INDEX_COL] == id, ESSAY_COL].iloc[0]
                encoded = tokenizer(text_full, return_tensors='pt', truncation=True, padding=True)
                input_ids = encoded['input_ids'].to(DEVICE)
                attention_mask = encoded['attention_mask'].to(DEVICE)

                rows = essays_df.loc[essays_df[INDEX_COL] == id, lingfeat_cols]
                if len(rows) > 1:
                    # if all rows are identical, keep one
                    if rows.eq(rows.iloc[0]).all().all():
                        rows = rows.iloc[[0]]
                    else:
                        # check rows without NaNs
                        rows_no_nan = rows.dropna()

                        if len(rows_no_nan) >= 1:
                            rows = rows_no_nan.iloc[[0]]
                        else:
                            # fallback: keep first row
                            rows = rows.iloc[[0]]
                else:
                    rows = rows.iloc[[0]]

                lingfeats = torch.tensor(rows.values, dtype=torch.float).to(DEVICE)
                pred_full_essay = model(input_ids, attention_mask, lingfeats).item()

                essay_preds[id] = pred_full_essay

            # Predict LOO essay
            text_loo = row['essay_loo']
            try:
                encoded_loo = tokenizer(text_loo, return_tensors='pt', truncation=True, padding=True)
                input_ids_loo = encoded_loo['input_ids'].to(DEVICE)
                attention_mask_loo = encoded_loo['attention_mask'].to(DEVICE)

                rows = sents_df[
                    (sents_df[INDEX_COL]==id) &
                    (sents_df[index_loo_col]==removed_indices)
                ][lingfeat_cols]
                if len(rows) > 1:

                    # if all rows are identical, keep one
                    if rows.eq(rows.iloc[0]).all().all():
                        rows = rows.iloc[[0]]
                    else:
                        # check rows without NaNs
                        rows_no_nan = rows.dropna()

                        if len(rows_no_nan) >= 1:
                            rows = rows_no_nan.iloc[[0]]
                        else:
                            # fallback: keep first row
                            rows = rows.iloc[[0]]
                else:
                    rows = rows.iloc[[0]]

                lingfeats = torch.tensor(rows.values, dtype=torch.float).to(DEVICE)
                pred_loo = model(input_ids_loo, attention_mask_loo, lingfeats).item()
                essay_loo_preds.append((id, removed_indices, pred_loo, essay_preds[id]))
            except ValueError:
                assert pd.isna(text_loo)
                essay_loo_preds.append((id, removed_indices, np.nan, essay_preds[id]))

    return pd.DataFrame(essay_loo_preds, columns=[INDEX_COL, index_loo_col, 'pred_loo', 'pred_essay'])

# multi loo predictions helper function
def merge_predictions_multi(window_size, sents_df, config):
    INDEX_COL = config.index_col
    TARGET_COL = config.target_col
    TASK_COL = config.task_col
    ESSAY_COL = config.essay_col

    # stringified lists to python lists
    sents_df['removed_indices'] = sents_df['removed_indices'].apply(ast.literal_eval)
    sents_df['removed_sentences'] = sents_df['removed_sentences'].apply(ast.literal_eval)

    # merge predictions considering window
    avg_pred_loo = []

    print("Merging multi predictions...")
    for id in tqdm(sents_df[INDEX_COL].unique()):
        df_id = sents_df[sents_df[INDEX_COL] == id].reset_index()
        all_indices = list(range(0, len(df_id)-(window_size-1)))

        for idx in all_indices:
            matching_rows = df_id[df_id['removed_indices'].apply(lambda x: idx in x)]
            removed_sentence = df_id.loc[idx, 'removed_sentences'][-1]
            pred_essay, task, true_score, full_essay = df_id.loc[idx, ['pred_essay', TASK_COL, TARGET_COL, ESSAY_COL]]
            avg_pred_loo.append([
                id, task, full_essay, removed_sentence, idx, matching_rows['pred_loo'].mean(), pred_essay, true_score
            ])

    # keep just predictions
    avg_pred_loo = pd.DataFrame(avg_pred_loo, columns=[
        INDEX_COL, TASK_COL, ESSAY_COL, 'sentence', 'sentence_index', 'pred_loo', 'pred_essay', TARGET_COL
    ])
    return avg_pred_loo

# MAIN FUNCTION
def run_ablation_hybrid(config):
    print("="*30+"\n")
    print("RUNNING ABLATION")

    # hyperparameters
    INPUT_FILENAME = config.input_filename
    INDEX_COL = config.index_col
    TASK_COL = config.task_col
    TARGET_COL = config.target_col
    ESSAY_COL = config.essay_col
    ABLATION = config.ablation
    INPUT_FILENAME = config.input_filename
    MODEL_NAME = config.model_name
    PROMPT = config.prompt

    if ABLATION == "single":
        print("\nSINGLE SENTENCE ABLATION\n")

        # checking if essays loo already predicted
        loo_predictions_path = os.path.join('loo_predictions', MODEL_NAME+'_loosingle.csv')
        if loo_predictions_exists(loo_predictions_path):
            return {}

        # loading loo essays preprocessed data
        filepath = os.path.join('data', 'processed', INPUT_FILENAME[:4]+f'_set{PROMPT}_loosingle_with_features.csv')
        sents_df = pd.read_csv(filepath)

        # loading full essays preprocessed data
        essays_df = pd.read_csv(os.path.join('data', INPUT_FILENAME))

        # predicting on loo essays
        loo_df = predict_essay_loo(sents_df, essays_df, 'sentence_index', config)

        # merging file
        sents_df = sents_df.merge(loo_df, on=[INDEX_COL, 'sentence_index'], how='left')

        # saving results
        print("Saving LOO predictions to:", loo_predictions_path)
        sents_df.to_csv(loo_predictions_path, index=False)

        return {}

    elif ABLATION in ["multi3", "multi5"]:
        window_size = config.window_size

        print("\nMULTI SENTENCE ABLATION:")
        print(f"WINDOW SIZE = {window_size}\n")

        # checking if essays loo already predicted
        loo_predictions_path = os.path.join('loo_predictions', MODEL_NAME+f'_loomulti_{window_size}.csv')
        if loo_predictions_exists(loo_predictions_path):
            return {}

        # loading preprocessed data
        filepath = os.path.join('data', 'processed', INPUT_FILENAME[:4]+f'_set{PROMPT}_loomulti_{window_size}_with_features.csv')
        sents_df = pd.read_csv(filepath)

        # loading full essays preprocessed data
        essays_df = pd.read_csv(os.path.join('data', INPUT_FILENAME))

        # predicting on loo essays
        loo_df = predict_essay_loo(sents_df, essays_df, 'removed_indices', config)

        # merging file
        sents_df = sents_df.merge(loo_df, on=[INDEX_COL, 'removed_indices'], how='left')
        sents_df = merge_predictions_multi(window_size, sents_df, config)

        # saving results
        print("Saving LOO predictions to:", loo_predictions_path)
        sents_df.to_csv(loo_predictions_path, index=False)

        return {}


if __name__ == "__main__":
    pass
