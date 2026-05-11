
import os
import pandas as pd
import numpy as np
import ast

from tqdm import tqdm

from essay_level_training import essay_lr


def loo_predictions_exists(filepath):
    if os.path.isfile(filepath):
        print("Loo essay predictions already computed!")
        print("File:", filepath)
        print("="*30+"\n")
        return True
    return False

# LOO AND FULL ESSAY PREDICTION
def predict_essay_loo(sents_df, essays_df, config):
    MODEL_NAME = config.model_name

    model_path = os.path.join('models', MODEL_NAME+'.pkl')

    # predict on LOO essays
    print(f"Predicting scores on {config.ablation} essay loo")
    pred_sents = essay_lr.predict_with_pipeline(model_path, sents_df)

    # predict on full essays
    print(f"Predicting scores on full essays")
    pred_essays = essay_lr.predict_with_pipeline(model_path, essays_df)

    return pred_sents, pred_essays

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
def run_ablation_lr(config):
    print("="*30+"\n")
    print("RUNNING ABLATION")

    # hyperparameters
    INPUT_FILENAME = config.input_filename
    ESSAY_COL = config.essay_col
    INDEX_COL = config.index_col
    TASK_COL = config.task_col
    TARGET_COL = config.target_col
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
        sents_df = pd.read_csv(os.path.join('data', 'processed', INPUT_FILENAME[:4]+f'_set{PROMPT}_loosingle_with_features.csv'))
        # loading full essays preprocessed data
        essays_df = pd.read_csv(os.path.join('data', INPUT_FILENAME))

        # predicting on loo essays
        sents_df['pred_loo'], pred_essays = predict_essay_loo(sents_df, essays_df, config)

        # sanity check
        assert len(pred_essays) == len(essays_df), "pred_essays length mismatch"

        pred_essays_df = pd.DataFrame({
            INDEX_COL: essays_df[INDEX_COL].values,
            'pred_essay': pred_essays
        })

        # merge loo and essay predictions
        sents_df = sents_df.merge(pred_essays_df, on=[INDEX_COL], how='left')

        # keep just predictions
        need_from_essays = [TASK_COL, ESSAY_COL, TARGET_COL]
        cols_to_merge = [INDEX_COL] + [c for c in need_from_essays if c not in sents_df.columns]

        if len(cols_to_merge) > 1:
            sents_df = sents_df.merge(
                essays_df[cols_to_merge],
                on=INDEX_COL,
                how="left"
            )
        sents_df = sents_df[[INDEX_COL, TASK_COL, ESSAY_COL, 'sentence', 'sentence_index', 'essay_loo', 'pred_loo', 'pred_essay', TARGET_COL]]

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

        # loading loo essays preprocessed data
        filepath = os.path.join('data', 'processed', INPUT_FILENAME[:4]+f'_set{PROMPT}_loomulti_{window_size}_with_features.csv')
        sents_df = pd.read_csv(filepath)
        # loading full essays preprocessed data
        essays_df = pd.read_csv(os.path.join('data', INPUT_FILENAME))

        # predicting on loo essays
        sents_df['pred_loo'], pred_essays = predict_essay_loo(sents_df, essays_df, config)

        # sanity check
        assert len(pred_essays) == len(essays_df), "pred_essays length mismatch"

        pred_essays_df = pd.DataFrame({
            INDEX_COL: essays_df[INDEX_COL].values,
            'pred_essay': pred_essays
        })

        sents_df = sents_df.merge(pred_essays_df, on=INDEX_COL, how='left')
        sents_df = merge_predictions_multi(window_size, sents_df, config)

        # saving results
        print("Saving LOO predictions to:", loo_predictions_path)
        sents_df.to_csv(loo_predictions_path, index=False)


if __name__ == "__main__":
    pass
