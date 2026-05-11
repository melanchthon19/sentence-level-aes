
import os
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit


def load_data(config):
    # ==== HYPERPARAMETERS ====
    INPUT_FILENAME = config.input_filename
    PROMPT = config.prompt
    TASK_COL = config.task_col
    BIN_COL = config.bin_col
    ESSAY_COL = config.essay_col
    INDEX_COL = config.index_col
    TARGET_COL = config.target_col
    RANDOM_SEED = config.random_seed

    # ==== DATA LOADING ====
    print("="*30+"\n")
    print("LOADING DATA")
    try:
        df = pd.read_csv(os.path.join('data', INPUT_FILENAME))
    except UnicodeDecodeError:
        df = pd.read_csv(os.path.join('data', INPUT_FILENAME), encoding="latin1", sep="\t")
    print("Initial data size:", df.shape)

    # ==== DATA FILTERING ====
    print("Filtering data:")
    if PROMPT == 'TEAD':
        pass # both prompts TE and AD
    else:
        df = df[df[TASK_COL]==PROMPT].copy()
        print("Data filtered!")

    before = len(df)
    df = df.dropna(subset=[TASK_COL, BIN_COL, ESSAY_COL, INDEX_COL, TARGET_COL])
    dropped = before - len(df)
    print(f"Dropped {dropped} rows due to missing values.")
    print("Data size:", df.shape)

    # ==== STRATIFIED TRAIN, VAL, TEST SPLITS ====
    binned = pd.cut(df[BIN_COL], bins=3, include_lowest=True, labels=["low", "medium", "high"])
    df.insert(0, "binned", binned)

    # First split: train_val (80%) and test (20%)
    strat_split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    for train_val_idx, test_idx in strat_split1.split(df, df['binned']):
        train_val_df = df.iloc[train_val_idx]
        test_df = df.iloc[test_idx]

    # Second split: train (70%) and val (10%) from the train_val (80%)
    strat_split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=RANDOM_SEED)
    for train_idx, val_idx in strat_split2.split(train_val_df, train_val_df['binned']):
        train_df = train_val_df.iloc[train_idx]
        val_df = train_val_df.iloc[val_idx]

    print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("\ntrain_df")
    print(train_df['binned'].value_counts().sort_index())

    print("\nval_df")
    print(val_df['binned'].value_counts().sort_index())

    print("\ntest_df")
    print(test_df['binned'].value_counts().sort_index())

    return df, {'train_df': train_df, 'val_df': val_df, 'test_df': test_df}

if __name__ == "__main__":
    pass
