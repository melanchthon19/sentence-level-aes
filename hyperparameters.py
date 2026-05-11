# DEFAULT HYPERPARAMETERS

# === LLM AND HYBRID HYPERPARAMETERS ===
BATCH_SIZE = 16
MAX_LEN = 512
EPOCHS = 20
LEARNING_RATE = 0.005
LOSS_FUNC = "MSELoss"
EARLY_STOPPING_PATIENCE = 5
FREEZE_PARAMS = True
DEVICE = "cuda"

# === TASK HYPERPARAMETERS ===
PROMPT = 1 # on asap, prompts are int, not str!
TARGET_COL = "domain1_score"

# === FILE HYPERPARAMETERS ===
INPUT_FILENAME = "asap_with_features.csv"

# === PSEUDO SCORE HYPERPARAMETERS ===
ABLATION = "single" # [single, multi3, multi5]
ALPHA = 1
ADD_BIAS = True
APPLY_NORM = "tanh" # [min_max, znorm, tanh, none]

# === MODEL ===
MODEL = "HYBRID"

# === LLM MODEL DIR ===
MODEL_DIR = "$WORK/.cache/huggingface/models--distilbert-base-uncased/snapshots/12040accade4e8a0f71eabdb258fecc2e7e948be/"

# === RREPLICABILITY ===
RANDOM_SEED = 42

def update_filespecific(input_filename):
    if input_filename == "MEWS_htiw_lingfeats.csv":
        return {
            'essay_col': "essay_normalized",
            'index_col': "id",
            'task_col': "task",
            'bin_col': "holistic",
        }
    elif input_filename == "MEWS_hiw_lingfeats.csv":
        return {
            'essay_col': "Text",
            'index_col': "Text_ID",
            'task_col': "Prompt",
            'bin_col': "Label",
        }
    elif "asap" in input_filename:
        return {
            'essay_col': "essay",
            'index_col': "essay_id",
            'task_col': "essay_set",
            'bin_col': "domain1_score",
        }
