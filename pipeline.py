
import os
import argparse
import csv, pathlib, datetime

import hyperparameters as hp

from load_data import load_data

from essay_level_training import essay_distilbert, essay_hybrid, essay_lr
from sentence_level_training import sentence_distilbert, sentence_hybrid, sentence_lr

from ablation import ablation_distilbert, ablation_hybrid, ablation_lr

from evaluate import evaluate_sentence_predictions


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline Hyperparameters")

    # Model
    parser.add_argument("--model", type=str, choices=["LLM", "HYBRID", "LR"], default=hp.MODEL)

    # LLM and HYBRID
    parser.add_argument("--batch_size", type=int, default=hp.BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=hp.MAX_LEN)
    parser.add_argument("--epochs", type=int, default=hp.EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=hp.LEARNING_RATE)
    parser.add_argument("--loss_func", type=str, default=hp.LOSS_FUNC)
    parser.add_argument("--early_stopping_patience", type=int, default=hp.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--freeze_params", type=bool, default=hp.FREEZE_PARAMS)
    parser.add_argument("--device", type=str, default=hp.DEVICE)
    parser.add_argument("--random_seed", type=int, default=hp.RANDOM_SEED)
    parser.add_argument("--model_dir", type=str, default=hp.MODEL_DIR)

    # Task
    parser.add_argument("--prompt", type=str, default=hp.PROMPT)
    parser.add_argument("--target_col", type=str, default=hp.TARGET_COL)

    # Data
    parser.add_argument("--input_filename", type=str, default=hp.INPUT_FILENAME)

    # Pseudo scores
    parser.add_argument("--alpha", type=float, default=hp.ALPHA)
    parser.add_argument("--add_bias", action="store_true", default=True) # defaults to true always
    parser.add_argument("--ablation", type=str, choices=["single", "multi3", "multi5"], default=hp.ABLATION)
    parser.add_argument("--apply_norm", type=str, choices=["min_max", "znorm", "tanh", "None"], default=hp.APPLY_NORM)
    parser.add_argument("--baseline", action="store_true", default=False)

    return parser.parse_args()

def save_results(config, results, out_csv):
    ts = datetime.datetime.now().isoformat(timespec="seconds")

    row = {
        "timestamp": ts,
        "model": config.model,
        "prompt": config.prompt,
        "target": config.target_col,
        "input_filename": config.input_filename,
        "alpha": config.alpha,
        "add_bias": config.add_bias,
        "ablation": config.ablation,
        "apply_norm": config.apply_norm,
        # metrics
        **results,
    }

    # CSV
    write_header = not pathlib.Path(out_csv).exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print("Results appended to:", out_csv)

def create_folders():
    os.makedirs("models", exist_ok=True)
    os.makedirs("essay_predictions", exist_ok=True)
    os.makedirs("loo_predictions", exist_ok=True)
    os.makedirs("sentence_predictions", exist_ok=True)
    os.makedirs("evaluation_predictions", exist_ok=True)

def update_hyperparameters(config):
    hps = hp.update_filespecific(config.input_filename)
    for k, v in hps.items():
        setattr(config, k.lower(), v)

    if "asap" in config.input_filename:
        config.prompt = int(config.prompt)
        
    setattr(config, 'model_name', f"{config.model}_{config.prompt}_{config.target_col}")

    if config.ablation == "multi3":
        setattr(config, 'window_size', 3)
    elif config.ablation == "multi5":
        setattr(config, 'window_size', 5)
    else:
        setattr(config, 'window_size', 1)

    return config

# MAIN FUNCTION
def pipeline(config):
    print("\nRunning with config:")
    config = update_hyperparameters(config)
    config_dict = vars(config)
    width = max(len(k) for k in config_dict.keys())
    for k, v in config_dict.items():
        print(f"  {k.ljust(width)} : {v}")

    create_folders()
    date = datetime.datetime.today().date().strftime("%d-%m")

    # STEP 1: load data
    df, df_sets = load_data(config)

    if config.model == "LLM":
        # STEP 2: train essay level model
        results = essay_distilbert.train_essay_distilbert(df, df_sets, config)
        save_results(config, results, out_csv=f"{config.model}_essayresults_{date}.csv")

        # STEP 3: generate pseudo labels
        results = ablation_distilbert.run_ablation_distilbert(config)

        # STEP 4: train sentence level model
        results = sentence_distilbert.train_sentence_distilbert(df, df_sets, config)


    if config.model == "LR":
        # STEP 2: train essay level model
        results = essay_lr.train_essay_lr(df, df_sets, config)
        save_results(config, results, out_csv=f"{config.model}_essayresults_{date}.csv")

        # STEP 3: generate pseudo labels
        results = ablation_lr.run_ablation_lr(config)

        # STEP 4: train sentence level model
        results = sentence_lr.train_sentence_lr(df, df_sets, config)

    if config.model == "HYBRID":
        # STEP 2: train essay level model
        results = essay_hybrid.train_essay_hybrid(df, df_sets, config)
        save_results(config, results, out_csv=f"{config.model}_essayresults_{date}.csv")

        # STEP 3: generate pseudo labels
        results = ablation_hybrid.run_ablation_hybrid(config)

        # STEP 4: train sentence level model
        results = sentence_hybrid.train_sentence_hybrid(df, df_sets, config)

    # STEP 5: evaluate on essay labels
    results = evaluate_sentence_predictions(df, df_sets, config)

    # Save results
    save_results(config, results, out_csv=f"{config.model}_sentenceresults_{date}.csv")


if __name__ == "__main__":
    args = parse_args()
    pipeline(args)
