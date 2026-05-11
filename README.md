# Explainable Essay Scoring through Fine-Grained Predictions

This repository contains the source code for the article [**"Through the Sentence Lens: Explainable Essay Scoring through Fine-Grained Predictions"**](https://github.com/melanchthon19/sentence-level-aes). This research proposes an ablation approach for training sentence-level Automated Essay Scoring (AES) models without requiring sentence-level annotations, and evaluates their performance against essay-level baselines.

The repository contains a modular training/evaluation pipeline for essay-level and sentence-level regression models (encoder-based, linguistic-based, and hybrid). The pipeline implements the following steps:
- Data loading
- Essay-level training and evaluation
- Generation of pseudo scores at sentence-level
- Sentence-level training on pseudo scores
- Evaluation of sentence-level model on essay-level gold labels

## Usage

- Install dependencies.
- Add your data to `data` directory.
- Adjust hyperparameters in `hyperparameters.py`
- Run main file `pipeline.py` passing any relevant argument through the command line.
- Note that if you run linguistic-based or hybrid model using your own linguistic features, the column names must be updated in their respective training files.

The `pipeline.py` file has three main steps:
1. Train essay-level model: This model is trained using gold labels, and is used as the baseline model.
2. Generate pseudo scores: This step uses the essay-level model to generate pseudo scores at sentence level through an ablation procedure. The result of this step is one pseudo score per sentence.
3. Train sentence-level model: Using the pseudo scores generated from step 2, a new sentence-level model is trained. This model will be compared to the essay-level baseline.
