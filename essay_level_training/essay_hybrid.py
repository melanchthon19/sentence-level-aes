
import os
import json
import pandas as pd
import numpy as np
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizerFast, DistilBertModel

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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


# ==== TOKENIZER ====
def load_tokenizer(model_dir):
    tokenizer = DistilBertTokenizerFast.from_pretrained(
                    os.path.expandvars(model_dir),
                    local_files_only=True
                    )
    return tokenizer

# ==== MODEL ====
class DistilBERTRegressor(nn.Module):
    def __init__(self, freeze_params, ling_feat_dim, model_dir):
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

        self.ling_mlp = nn.Sequential(
            nn.Linear(ling_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(768 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, lingfeats):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        pooled = self.pool(last_hidden.permute(0, 2, 1)).squeeze(-1)
        ling_out = self.ling_mlp(lingfeats)
        combined = torch.cat([pooled, ling_out], dim=1)
        pred = self.regressor(combined).squeeze(-1)
        return pred

def load_model(model_path, ling_feat_dim, model_dir):
    print(f"Loading essay level model from {model_path}")
    model = DistilBERTRegressor(freeze_params=True, ling_feat_dim=ling_feat_dim, model_dir=model_dir)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except RuntimeError:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    return model

def train_essay_hybrid(df, df_sets, config):
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

    if config.device == "mps" and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif config.device == "cuda" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("="*30+"\n")
    print("TRAINING ESSAY LEVEL DISTILBERT")

    # fix random
    fix_random(RANDOM_SEED)

    # data
    train_df = df_sets['train_df'].copy()
    val_df = df_sets['val_df'].copy()
    test_df = df_sets['test_df'].copy()
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # save trained model to model_path
    model_path = os.path.join('models', MODEL_NAME+'.pt')
    scaler_path = os.path.join('models', MODEL_NAME+'_scaler.pkl')

    # save predictions to predictions_path
    predictions_path = os.path.join('essay_predictions', MODEL_NAME+'.csv')

    # min and max values
    min_possible_value = df[TARGET_COL].min()
    max_possible_value = df[TARGET_COL].max()
    print("Min and max values:", min_possible_value, max_possible_value)

    start_idx = df.columns.get_loc("count_words")
    lingfeat_cols = list(df.columns[start_idx:start_idx+220])
    if len(lingfeat_cols) != 220:
        raise ValueError(f"Linguistic features dimension is not 220, but {len(lingfeat_cols)}")

    # preprocessing lingfeat pipeline
    scaler = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", MinMaxScaler()),
    ])

    # imputing and scaling features
    print("Imputing (mean) and scaling linguistic features...")
    scaler.fit(train_df[lingfeat_cols])
    train_df[lingfeat_cols] = scaler.transform(train_df[lingfeat_cols]).copy()
    val_df[lingfeat_cols] = scaler.transform(val_df[lingfeat_cols]).copy()
    test_df[lingfeat_cols] = scaler.transform(test_df[lingfeat_cols]).copy()

    joblib.dump(scaler, scaler_path)
    print("Scaler saved to", scaler_path)

    tokenizer = load_tokenizer(MODEL_DIR)

    # ==== DATASET ====
    class EssayDataset(Dataset):
        def __init__(self, texts, lingfeats, targets, ids, idxs):
            self.texts = list(texts)
            self.lingfeats = torch.tensor(lingfeats.values, dtype=torch.float)
            self.targets = torch.tensor(targets.values, dtype=torch.float)
            self.ids = ids.values
            self.idxs = idxs.values

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
                'lingfeats': self.lingfeats[idx],
                'essay_ids': self.ids[idx],
                'essay_idxs': self.idxs[idx]
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
                lingfeats = batch['lingfeats'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                preds = model(input_ids, attention_mask, lingfeats)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for b_idx, batch in enumerate(val_loader):#batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    lingfeats = batch['lingfeats'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)

                    preds = model(input_ids, attention_mask, lingfeats)

                    # check nan or inf
                    preds_cpu = preds.detach().cpu()
                    labels_cpu = labels.detach().cpu()

                    n_pred_nan = torch.isnan(preds_cpu).sum().item()
                    n_pred_inf = torch.isinf(preds_cpu).sum().item()
                    n_lab_nan  = torch.isnan(labels_cpu).sum().item()
                    n_lab_inf  = torch.isinf(labels_cpu).sum().item()

                    if n_pred_nan or n_pred_inf or n_lab_nan or n_lab_inf:
                        print(
                            f"[VAL BATCH {b_idx}] "
                            f"preds NaN={n_pred_nan} Inf={n_pred_inf} | "
                            f"labels NaN={n_lab_nan} Inf={n_lab_inf}"
                        )
                        print("preds:", preds_cpu[:5])
                        print("labels:", labels_cpu[:5])
                        raise RuntimeError("NaN/Inf detected in validation batch")

                    # end check nan or inf

                    val_preds.append(preds_cpu.numpy())
                    val_labels.append(labels_cpu.numpy())

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
                torch.save(model.state_dict(), model_path)
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
    train_dataset = EssayDataset(train_df[ESSAY_COL], train_df[lingfeat_cols], train_df[TARGET_COL], train_df[INDEX_COL], train_df.index)
    val_dataset = EssayDataset(val_df[ESSAY_COL], val_df[lingfeat_cols], val_df[TARGET_COL], val_df[INDEX_COL], val_df.index)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = DistilBERTRegressor(freeze_params=FREEZE_PARAMS, ling_feat_dim=len(lingfeat_cols), model_dir=MODEL_DIR).to(DEVICE)

    # Checking if model already trained
    if os.path.isfile(model_path):
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
    test_dataset = EssayDataset(test_df[ESSAY_COL], test_df[lingfeat_cols], test_df[TARGET_COL], test_df[INDEX_COL], test_df.index)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = DistilBERTRegressor(freeze_params=FREEZE_PARAMS, ling_feat_dim=len(lingfeat_cols), model_dir=MODEL_DIR).to(DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    test_preds, test_labels, test_essay_ids, test_essay_idxs = [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            lingfeats = batch['lingfeats'].to(DEVICE)
            essay_ids = batch['essay_ids']
            essay_idxs = batch['essay_idxs']

            preds = model(input_ids, attention_mask, lingfeats)
            test_preds.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_essay_ids.extend(essay_ids)
            test_essay_idxs.extend(essay_idxs.cpu().numpy())

    test_preds = np.clip(np.concatenate(test_preds), min_possible_value, max_possible_value)
    test_labels = np.concatenate(test_labels)

    essay_preds = pd.DataFrame(data={
        'index': test_essay_idxs,
        'prompt': PROMPT,
        'target': TARGET_COL,
        'essay_id': test_essay_ids,
        'prediction': test_preds,
        'true_score': test_labels,
    })

    essay_preds.to_csv(predictions_path, index=False)
    print("Predictions saved to:", predictions_path)

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
