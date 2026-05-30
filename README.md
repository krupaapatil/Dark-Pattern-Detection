# Dark Patterns Data Mining Project

This is a simple Python project that compares multiple algorithms on dark-pattern and scam-text datasets, then adds a lightweight category-explanation layer for manipulative text.

## What it does

- reads `dataset.tsv`, a synthetic dark-pattern dataset, and `scam_dataset.tsv`
- uses `text` and `label` columns
- creates separate training and testing data for each dataset
- merges the splits into one combined training and testing set
- trains 5 algorithms:
  - Logistic Regression
  - Linear SVM
  - SGD Log Loss
  - Complement Naive Bayes
  - Hybrid Stacking
- compares them using:
  - accuracy
  - precision
- saves the best binary model
- trains a second-stage category model that predicts the likely manipulation type such as Urgency, Scarcity, Social Proof, or Scam / Phishing

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python dark_patterns_project.py
```

To open the text-checking interface:

```bash
python dark_patterns_interface.py
```

Paste text into the box and the app will predict whether it looks manipulative/dark-pattern-like or normal, and if it is manipulative it will also show the likely pattern category.

## Output

After running, check the `outputs/` folder:

- `original_train_split.csv`
- `original_test_split.csv`
- `synthetic_train_split.csv`
- `synthetic_test_split.csv`
- `scam_train_split.csv`
- `scam_test_split.csv`
- `train_split.csv`
- `test_split.csv`
- `model_results.csv`
- `best_model.joblib`
- `category_model_results.csv`
- `best_category_model.joblib`

## Simple project flow

1. Load the original, synthetic, and scam datasets
2. Split each dataset into training and testing data
3. Merge the training splits and testing splits
4. Train and compare the binary models
5. Train a category model on manipulative examples only
6. Use the two-stage output for prediction plus explanation
