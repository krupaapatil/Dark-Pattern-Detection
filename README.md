# Dark Pattern Text Analyzer

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Project](https://img.shields.io/badge/Project-Data%20Mining-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A Python machine learning project that detects manipulative, dark-pattern, scam-like, or phishing-style text and explains the likely manipulation category through a simple desktop interface.

This is an educational text-only ML project. It is useful for learning data mining, text classification, and model comparison, but it should not be treated as a production fraud-detection system.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python dark_patterns_interface.py
```

The repository includes trained model files in `outputs/`, so the interface can run without retraining first.

## Highlights

- Binary text classification for normal vs manipulative wording
- Category prediction for dark-pattern style explanations
- Comparison of five machine learning algorithms
- TF-IDF based text features with scikit-learn pipelines
- Reproducible train/test split generation
- Desktop interface for checking custom messages
- PDF report generator for project documentation

## Tech Stack

- Python
- pandas
- scikit-learn
- joblib
- Tkinter
- ReportLab
- openpyxl

## Model Results

The best binary classifier from the saved run is `Linear SVM`. It achieved the strongest overall F1 score while keeping precision high, which makes it a good fit for identifying risky or manipulative wording without over-labeling normal text.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Linear SVM | 0.9604 | 0.9831 | 0.9544 | 0.9685 |
| Hybrid Stacking | 0.9592 | 0.9741 | 0.9617 | 0.9679 |
| SGD Log Loss | 0.9592 | 0.9812 | 0.9544 | 0.9676 |
| Logistic Regression | 0.9487 | 0.9865 | 0.9325 | 0.9587 |
| Complement Naive Bayes | 0.9441 | 0.9513 | 0.9617 | 0.9564 |

The category model uses `Linear SVM` and predicts 8 manipulation categories with `0.9617` accuracy on the saved test split.

## Project Structure

```text
.
|-- dark_patterns_project.py          # Training and evaluation pipeline
|-- dark_patterns_interface.py        # Tkinter prediction interface
|-- project_report_generator.py       # PDF report generator
|-- dataset.tsv                       # Original labeled dataset
|-- scam_dataset.tsv                  # Scam/phishing examples
|-- synthetic_dark_pattern_dataset_2500.xlsx
|-- outputs/
|   |-- best_model.joblib
|   |-- best_category_model.joblib
|   |-- model_results.csv
|   `-- category_model_results.csv
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Training Pipeline

```bash
python dark_patterns_project.py
```

This creates train/test splits, evaluates all models, and saves the best binary and category models in `outputs/`.

## Open the Interface

```bash
python dark_patterns_interface.py
```

Paste any sentence, ad copy, alert, or suspicious message into the app. The interface returns:

- whether the text appears normal or manipulative
- the likely manipulation category, when detected
- a short explanation of the detected pattern

## Generate the PDF Report

```bash
python project_report_generator.py
```

The report is saved as `dark_patterns_project_report.pdf`.

## Categories

The app can explain detected manipulative text with categories such as:

- Urgency
- Scarcity
- Social Proof
- Misdirection
- Obstruction
- Sneaking
- Forced Action
- Scam / Phishing

## Limitations

This project only analyzes text. It does not inspect website layout, colors, button placement, checkout flows, screenshots, or user behavior. Predictions depend on the included datasets and should be interpreted as model outputs, not confirmed security judgments.

## Future Improvements

- Add a web interface with Streamlit or Flask
- Add confidence scores for predictions
- Expand the dataset with real-world dark-pattern examples
- Include visual and layout-based dark-pattern detection
- Add cross-validation and confusion matrix visualizations
