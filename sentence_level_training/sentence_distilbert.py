
import os
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, Normalizer

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizerFast, DistilBertModel
from sklearn.metrics import mean_squared_error, r2_score, cohen_kappa_score
from scipy.stats import pearsonr

from tqdm import tqdm


def fix_random(RANDOM_SEED):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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

# ==== TOKENIZER ====
def load_tokenizer(model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(
                    os.path.expandvars(model_dir),
                    local_files_only=True
                )
    return tokenizer

# ==== MODEL ====
class DistilBERTRegressor(nn.Module):
    def __init__(self, freeze_params, model_dir):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(
                os.path.expandvars(model_dir),
                local_files_only=True
            )

        if freeze_params:
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        pooled = self.pool(last_hidden.permute(0, 2, 1)).squeeze(-1)
        pred = self.regressor(pooled).squeeze(-1)
        return pred

# MAIN FUNCTION
def train_sentence_distilbert(df, df_sets, config):
    # ==== HYPERPARAMETERS ====
    RANDOM_SEED = config.random_seed

    PROMPT = config.prompt
    TARGET_COL = config.target_col
    ESSAY_COL = config.essay_col
    INDEX_COL = config.index_col

    BATCH_SIZE = config.batch_size
    MAX_LEN = config.max_len
    EPOCHS = config.epochs
    LEARNING_RATE = config.learning_rate
    LOSS_FUNC = getattr(nn, config.loss_func)()
    EARLY_STOPPING_PATIENCE = config.early_stopping_patience
    FREEZE_PARAMS = config.freeze_params
    MODEL_DIR = config.model_dir
    MODEL_NAME = config.model_name
    ALPHA = config.alpha
    ADD_BIAS = config.add_bias
    APPLY_NORM = config.apply_norm
    ABLATION = config.ablation
    BASELINE = config.baseline

    window_size = config.window_size
    sentence_file = os.path.join('data', 'processed', config.input_filename[:-4]+'_loosingle.csv')

    if config.device == "mps" and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif config.device == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    if config.window_size == 1:
        loo_predictions_path = os.path.join('loo_predictions', MODEL_NAME+'_loosingle.csv')
    else:
        loo_predictions_path = os.path.join('loo_predictions', MODEL_NAME+f'_loomulti_{window_size}.csv')

    min_value = df[TARGET_COL].min()
    max_value = df[TARGET_COL].max()

    print("="*30+"\n")
    print("TRAINING SENTENCE LEVEL DISTILBERT")

    print("Loading initial dataset:", config.input_filename)
    df_full = pd.read_csv(os.path.join("data", config.input_filename))

    print("Loading loo predictions:", loo_predictions_path)
    dfps = pd.read_csv(loo_predictions_path)
    s = df_full.drop_duplicates(INDEX_COL).set_index(INDEX_COL)[TARGET_COL]
    dfps[TARGET_COL] = dfps[INDEX_COL].map(s)
    dfps = dfps.dropna(subset=["pred_loo", "pred_essay", TARGET_COL]).copy()
    dfps = dfps[np.isfinite(dfps["pred_loo"])].copy()

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
    sentence_model_path = os.path.join('models', sentence_model_name+'.pt')

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
    print(f"Essay level --> Train: {len(train_index)} | Val: {len(val_index)} | Test: {len(test_index)}")

    # filtering splits by ids SENTENCE LEVEL
    train_df = dfps[dfps[INDEX_COL].isin(train_index)]
    val_df = dfps[dfps[INDEX_COL].isin(val_index)]
    test_df = dfps[dfps[INDEX_COL].isin(test_index)]
    print(f"Sentence level --> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    assert np.isfinite(train_df["pseudo_scores"]).all()
    assert np.isfinite(val_df["pseudo_scores"]).all()
    assert np.isfinite(test_df["pseudo_scores"]).all()

    tokenizer = load_tokenizer(MODEL_DIR)

    # ==== DATASET ====
    class EssaySentenceDataset(Dataset):
        def __init__(self, texts, targets, ids, sent_idxs):
            self.texts = list(texts)
            self.targets = torch.tensor(targets.values, dtype=torch.float)
            self.ids = ids.values
            self.sent_idxs = sent_idxs.values

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            encoding = tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': self.targets[idx],
                'essay_ids': self.ids[idx],
                'sent_idxs': self.sent_idxs[idx],
            }

    # ==== TRAINING FUNCTION ====
    def train(model, train_loader, val_loader):
        print("Starting training...")
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = LOSS_FUNC
        best_val_loss = float('inf')
        best_val_pearson_r, best_val_pearson_p = -float("inf"), 1.0
        best_val_qwk, best_val_qwk_rescaled = -float("inf"), -float("inf")
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)

                    preds = model(input_ids, attention_mask)
                    val_preds.append(preds.cpu().numpy())
                    val_labels.append(labels.cpu().numpy())

            # RMSE
            val_preds = np.clip(np.concatenate(val_preds), min_possible_value, max_possible_value)
            val_labels = np.concatenate(val_labels)
            val_loss = np.sqrt(mean_squared_error(val_labels, val_preds))

            # PEARSON
            val_pearson_r, val_pearson_p = pearsonr(val_labels, val_preds)

            # QWK ROUNDED
            val_preds_rounded = np.round(np.clip(val_preds, min_possible_value, max_possible_value)).astype(int)
            val_labels_rounded = np.round(val_labels).astype(int)
            val_qwk = cohen_kappa_score(val_labels_rounded, val_preds_rounded, weights="quadratic")

            # QWK RESCALED
            val_preds_rescaled = np.round(np.clip(val_preds*2, min_possible_value*2, max_possible_value*2)).astype(int)
            val_labels_rescaled = np.round(val_labels*2).astype(int)
            val_qwk_rescaled = cohen_kappa_score(val_labels_rescaled, val_preds_rescaled, weights="quadratic")

            print(
                f"Epoch {epoch+1}: "
                f"Train Loss = {avg_train_loss:.4f} | "
                f"Val RMSE = {val_loss:.4f} | "
                f"Val Pearson r = {val_pearson_r:.4f} (p={val_pearson_p:.4f}) | "
                f"Val QWK Rounded = {val_qwk:.4f} | "
                f"Val QWK Rescaled = {val_qwk_rescaled:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_pearson_r, best_val_pearson_p = val_pearson_r, val_pearson_p
                best_val_qwk = val_qwk
                best_val_qwk_rescaled = val_qwk_rescaled
                patience_counter = 0
                torch.save(model.state_dict(), sentence_model_path)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping!")
                    break

        return {
            "best_val_loss": best_val_loss,
            "best_val_qwk": best_val_qwk,
            "best_val_qwk_rescaled": best_val_qwk_rescaled,
            "best_val_pearson_r": best_val_pearson_r,
            "best_val_pearson_p": best_val_pearson_p,
        }


    # ==== TRAINING ====
    train_dataset = EssaySentenceDataset(
        train_df['sentence'],
        train_df['pseudo_scores'],
        train_df[INDEX_COL],
        train_df['sentence_index']
    )

    val_dataset = EssaySentenceDataset(
        val_df['sentence'],
        val_df['pseudo_scores'],
        val_df[INDEX_COL],
        val_df['sentence_index']
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DistilBERTRegressor(freeze_params=FREEZE_PARAMS, model_dir=MODEL_DIR).to(DEVICE)

    # Checking if model already trained
    if os.path.isfile(sentence_model_path):
        print("Model already trained! Skipping training")
    else:
        results = train(model, train_loader, val_loader)
        print(
            f"Validation metrics: "
            f"RMSE: {results['best_val_loss']:.4f} | "
            f"QWK: {results['best_val_qwk']:.4f} | "
            f"QWK Rescaled: {results['best_val_qwk_rescaled']:.4f} | "
            f"Pearson r: {results['best_val_pearson_r']:.4f} "
            f"(p={results['best_val_pearson_p']:.4f})"
        )

    # ==== FINAL EVALUATION ON TEST SET ====
    print("\n===== Final Evaluation on Test Set =====")
    test_dataset = EssaySentenceDataset(
        test_df['sentence'],
        test_df['pseudo_scores'],
        test_df[INDEX_COL],
        test_df['sentence_index']
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DistilBERTRegressor(freeze_params=FREEZE_PARAMS, model_dir=MODEL_DIR).to(DEVICE)
    model.load_state_dict(torch.load(sentence_model_path, weights_only=True))
    model.eval()

    test_preds, test_labels, test_essay_ids, test_sent_idxs = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            essay_ids = batch['essay_ids']
            sent_idxs = batch['sent_idxs']

            preds = model(input_ids, attention_mask)
            test_preds.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_essay_ids.extend(essay_ids.cpu().numpy())
            test_sent_idxs.extend(sent_idxs.cpu().numpy())

    test_preds = np.clip(np.concatenate(test_preds), min_possible_value, max_possible_value)
    test_labels = np.concatenate(test_labels)

    essay_preds = pd.DataFrame(data={
        INDEX_COL: test_essay_ids,
        'sent_idx': test_sent_idxs,
        'prompt': PROMPT,
        'target': TARGET_COL,
        'prediction': test_preds,
        'pseudo_score': test_labels,
    })

    essay_preds.to_csv(sentence_predictions_path, index=False)
    print("Predictions saved to:", sentence_predictions_path)

    # METRICS
    rmse = np.sqrt(mean_squared_error(test_labels, test_preds))
    r2 = r2_score(test_labels, test_preds)
    pearson_r, pearson_p = pearsonr(test_labels, test_preds)

    # QWK ROUNDED
    test_preds_rounded = np.round(np.clip(test_preds, min_possible_value, max_possible_value)).astype(int)
    test_labels_rounded = np.round(test_labels).astype(int)
    qwk = cohen_kappa_score(test_labels_rounded, test_preds_rounded, weights="quadratic")

    # QWK RESCALED
    test_preds_rescaled = np.round(np.clip(test_preds*2, min_possible_value*2, max_possible_value*2)).astype(int)
    test_labels_rescaled = np.round(test_labels*2).astype(int)
    qwk_rescaled = cohen_kappa_score(test_labels_rescaled, test_preds_rescaled, weights="quadratic")

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


if __name__ == "__main__":
    pass
