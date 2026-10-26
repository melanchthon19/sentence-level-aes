# Explainable Essay Scoring through Fine-Grained Predictions

This repository contains the source code for the article [**"Through the Sentence Lens: Explainable Essay Scoring through Fine-Grained Predictions"**](https://aclanthology.org/2026.bea-1.16/). This research proposes fine-grained prediction as a means of enhancing the explainability of Automated Essay Scoring (AES) models. To address the lack of manual sentence-level annotations, we propose ablation strategies to generate pseudo scores for training sentence-level models. We evaluate their effectiveness against essay-level baseline.

## Usage

The repository contains a modular training/evaluation pipeline for essay-level and sentence-level regression models (encoder-based, linguistic-based, and hybrid).

- Install dependencies.
- Add your data to `data` directory.
- Adjust hyperparameters in `hyperparameters.py`
- Run main file `pipeline.py` passing any relevant argument through the command line.
- Note that if you run linguistic-based or hybrid model using your own linguistic features, the column names must be updated in their respective training files.

The `pipeline.py` file has three main steps:
1. Train essay-level model: This model is trained using gold labels, and is used as the baseline model.
2. Generate pseudo scores: This step uses the essay-level model to generate pseudo scores at sentence level through an ablation procedure. The result of this step is one pseudo score per sentence.
3. Train sentence-level model: Using the pseudo scores generated from step 2, a new sentence-level model is trained. This model will be compared to the essay-level baseline trained using gold labels.

## Bibtex

If you find this work useful, please consider citing:

```
@inproceedings{mora-etal-2026-sentence,
    title = "Through the Sentence Lens: Explainable Essay Scoring through Fine-Grained Predictions",
    author = "Mora, Daniel  and
      Keller, Stefan  and
      Horbach, Andrea",
    editor = "Kochmar, Ekaterina  and
      Alhafni, Bashar  and
      Bann{\`o}, Stefano  and
      Bexte, Marie  and
      Burstein, Jill  and
      Horbach, Andrea  and
      Laarmann-Quante, Ronja  and
      Tack, Anais  and
      Yaneva, Victoria  and
      Yuan, Zheng",
    booktitle = "Proceedings of the 21st Workshop on Innovative Use of {NLP} for Building Educational Applications ({BEA} 2026)",
    month = jul,
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.bea-1.16/",
    doi = "10.18653/v1/2026.bea-1.16",
    pages = "221--233",
    ISBN = "979-8-89176-409-5"
}
```
