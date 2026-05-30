from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORT_PATH = BASE_DIR / "dark_patterns_project_report.pdf"
RESULT_SORT_COLUMNS = ["f1", "accuracy", "precision"]
CATEGORY_RESULTS_FILE = "category_model_results.csv"


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def load_csv(file_name: str) -> pd.DataFrame:
    file_path = OUTPUTS_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}. Run the training script first.")
    return pd.read_csv(file_path)


def load_results() -> pd.DataFrame:
    return load_csv("model_results.csv")


def load_category_results() -> pd.DataFrame | None:
    file_path = OUTPUTS_DIR / CATEGORY_RESULTS_FILE
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def load_split_summary() -> dict[str, object]:
    train_df = load_csv("train_split.csv")
    test_df = load_csv("test_split.csv")
    original_train = load_csv("original_train_split.csv")
    original_test = load_csv("original_test_split.csv")
    synthetic_train = load_csv("synthetic_train_split.csv")
    synthetic_test = load_csv("synthetic_test_split.csv")
    scam_train = load_csv("scam_train_split.csv")
    scam_test = load_csv("scam_test_split.csv")

    original_total = len(original_train) + len(original_test)
    synthetic_total = len(synthetic_train) + len(synthetic_test)
    scam_total = len(scam_train) + len(scam_test)
    total_rows = len(train_df) + len(test_df)

    merged_labels = pd.concat([train_df["label"], test_df["label"]], ignore_index=True)
    label_counts = merged_labels.value_counts().to_dict()

    return {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "total_rows": total_rows,
        "original_train_rows": len(original_train),
        "original_test_rows": len(original_test),
        "synthetic_train_rows": len(synthetic_train),
        "synthetic_test_rows": len(synthetic_test),
        "scam_train_rows": len(scam_train),
        "scam_test_rows": len(scam_test),
        "original_total_rows": original_total,
        "synthetic_total_rows": synthetic_total,
        "scam_total_rows": scam_total,
        "dark_pattern_rows": int(label_counts.get(1, 0)),
        "not_dark_pattern_rows": int(label_counts.get(0, 0)),
    }


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="MyTitle",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MyHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1f3c88"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MyBody",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="MySmall",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=5,
        )
    )
    return styles


def add_paragraphs(story, styles, title, paragraphs):
    story.append(Paragraph(title, styles["MyHeading"]))
    for text in paragraphs:
        story.append(Paragraph(text, styles["MyBody"]))
    story.append(Spacer(1, 0.10 * inch))


def build_results_table(results: pd.DataFrame):
    table_data = [["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"]]
    for _, row in results.iterrows():
        table_data.append(
            [
                row["model"],
                format_metric(row["accuracy"]),
                format_metric(row["precision"]),
                format_metric(row["recall"]),
                format_metric(row["f1"]),
            ]
        )

    table = Table(
        table_data,
        colWidths=[2.4 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3c88")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def create_report():
    results = load_results().sort_values(by=RESULT_SORT_COLUMNS, ascending=False)
    category_results = load_category_results()
    best = results.iloc[0]
    summary = load_split_summary()
    styles = build_styles()

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    story = []
    story.append(Paragraph("Dark Patterns Data Mining Project Report", styles["MyTitle"]))
    story.append(
        Paragraph(
            "A three-dataset hierarchical text-classification study with binary detection and category explanation",
            styles["MySmall"],
        )
    )
    story.append(Spacer(1, 0.18 * inch))

    add_paragraphs(
        story,
        styles,
        "1. Introduction",
        [
            "This project studies dark patterns in e-commerce text. Dark patterns are messages or interface designs that try to push users into actions they may not fully want, such as buying quickly or making a rushed decision.",
            "The goal of this project is to use Python and machine learning to first detect whether a text is manipulative and then explain the likely manipulation type using dark-pattern categories.",
        ],
    )

    add_paragraphs(
        story,
        styles,
        "2. Scope Of The Project",
        [
            "The project is limited to text-based detection. It does not analyse full website images, page layout, colors, or user behavior.",
            "The main work is to clean three datasets, split them into training and testing sets, merge the split files, train multiple binary classifiers, and attach a lightweight category classifier for explainable output.",
        ],
    )

    add_paragraphs(
        story,
        styles,
        "3. Dataset Used",
        [
            f"The cleaned combined dataset contains {summary['total_rows']} text records after removing missing values and duplicate text-label pairs.",
            f"The original dataset contributes {summary['original_total_rows']} records, the synthetic dataset contributes {summary['synthetic_total_rows']} records, and the scam dataset contributes {summary['scam_total_rows']} records.",
            f"The binary labels are 1 for dark pattern and 0 for not dark pattern. In the merged cleaned data, there are {summary['dark_pattern_rows']} dark-pattern rows and {summary['not_dark_pattern_rows']} not-dark-pattern rows.",
            f"For this project, {summary['train_rows']} records were used for training and {summary['test_rows']} records were used for testing.",
        ],
    )

    add_paragraphs(
        story,
        styles,
        "4. Tools And Technologies",
        [
            "Programming language: Python",
            "Main libraries: pandas, scikit-learn, joblib, openpyxl, reportlab",
            "Text processing method: TF-IDF vectorization with unigram and bigram features",
            "Binary algorithms tested: Linear SVM, Logistic Regression, SGD Log Loss, Complement Naive Bayes, and a hybrid stacking model",
            "Explainability layer: a second Linear SVM predicts the likely manipulation category for texts already flagged as manipulative",
        ],
    )

    add_paragraphs(
        story,
        styles,
        "5. How The System Works",
        [
            "First, the project loads the original TSV dataset, the synthetic Excel dataset, and the scam TSV dataset using the same text and label schema.",
            "Next, each dataset is cleaned separately by selecting the needed columns, removing missing values, and removing duplicate text-label rows.",
            f"After that, each dataset is split into training data and testing data using an 80:20 ratio with a fixed random state. The original dataset produces {summary['original_train_rows']} training rows and {summary['original_test_rows']} testing rows, the synthetic dataset produces {summary['synthetic_train_rows']} training rows and {summary['synthetic_test_rows']} testing rows, and the scam dataset produces {summary['scam_train_rows']} training rows and {summary['scam_test_rows']} testing rows.",
            "Then, the three training splits are merged together and the three testing splits are merged together.",
            "The merged training data is used to train each algorithm. The merged testing data is kept separate and used only for final evaluation.",
            "Finally, the project compares the models using F1 score, accuracy, precision, and recall, and then trains a second-stage category classifier on the manipulative texts only.",
        ],
    )

    story.append(Paragraph("6. Pseudocode", styles["MyHeading"]))
    pseudocode = """START
Load original dataset
Load synthetic dataset
Load scam dataset

FOR each dataset
    Select shared columns
    Clean missing and duplicate text-label rows
    Split into training set and testing set
END FOR

Merge all training data
Merge all testing data

FOR each algorithm in [Linear SVM, Logistic Regression, SGD Log Loss, Complement Naive Bayes, Hybrid Stacking]
    Convert training text into TF-IDF features
    Train algorithm using merged training data
    Predict labels on merged testing data
    Calculate accuracy, precision, recall, and F1 score
END FOR

Compare all results using F1 score first
Choose the best model
Filter manipulative texts from the merged train and test data
Train a category classifier on manipulative texts only
Save the best model and result files
END"""
    story.append(Preformatted(pseudocode, styles["Code"]))
    story.append(Spacer(1, 0.12 * inch))

    add_paragraphs(
        story,
        styles,
        "7. Algorithms Used",
        [
            "Linear SVM is strong for text classification because it handles high-dimensional TF-IDF features very well.",
            "Logistic Regression is a reliable binary classification model and gives interpretable linear decision boundaries.",
            "SGD Log Loss is a fast linear classifier that works well on large sparse text features.",
            "Complement Naive Bayes is a useful probabilistic baseline for imbalanced text data.",
            "The hybrid stacking model combines Logistic Regression, a calibrated Linear SVM, and Complement Naive Bayes to test whether an ensemble can improve the final prediction.",
        ],
    )

    story.append(Paragraph("8. Findings And Results", styles["MyHeading"]))
    story.append(
        Paragraph(
            "The following table shows the test results of the five algorithms on the merged testing data.",
            styles["MyBody"],
        )
    )
    story.append(build_results_table(results))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            f"The best algorithm in this project is <b>{best['model']}</b>. It achieved an F1 score of <b>{format_metric(best['f1'])}</b>, an accuracy of <b>{format_metric(best['accuracy'])}</b>, a precision of <b>{format_metric(best['precision'])}</b>, and a recall of <b>{format_metric(best['recall'])}</b> on the merged test set.",
            styles["MyBody"],
        )
    )
    story.append(
        Paragraph(
            "The best model is selected by F1 score first, then accuracy, and then precision if a tie happens.",
            styles["MyBody"],
        )
    )
    if category_results is not None and not category_results.empty:
        category_best = category_results.iloc[0]
        story.append(
            Paragraph(
                f"The added explanation layer uses a secondary <b>{category_best['model']}</b> classifier to predict the likely manipulation category for positive texts. It achieved an accuracy of <b>{format_metric(category_best['accuracy'])}</b>, a macro F1 score of <b>{format_metric(category_best['macro_f1'])}</b>, and a weighted F1 score of <b>{format_metric(category_best['weighted_f1'])}</b> across <b>{int(category_best['category_count'])}</b> manipulation categories.",
                styles["MyBody"],
            )
        )
    story.append(
        Paragraph(
            "This makes the system hierarchical: it not only flags manipulative text, but also explains the likely pattern type such as urgency, scarcity, social proof, or scam/phishing.",
            styles["MyBody"],
        )
    )
    story.append(Spacer(1, 0.10 * inch))

    add_paragraphs(
        story,
        styles,
        "9. Conclusion",
        [
            "This project shows that machine learning can be used to detect dark-pattern text with strong performance even when training data comes from three different sources.",
            "Cleaning the datasets separately and merging the train and test splits gives a fairer evaluation than mixing duplicate rows across both stages.",
            f"Among the tested algorithms, {best['model']} gave the best overall result on the merged dataset.",
            "The added category layer gives the project a simple but useful novelty: the system can classify the text and also explain what kind of manipulation it resembles.",
        ],
    )

    add_paragraphs(
        story,
        styles,
        "10. Future Improvement",
        [
            "More algorithms can be tested, such as transformer models like BERT or RoBERTa.",
            "More text preprocessing can be tried, such as lemmatization, feature selection, or hyperparameter tuning.",
            "The category layer can be strengthened further with more rare-category samples and a larger external validation dataset.",
        ],
    )

    add_paragraphs(
        story,
        styles,
        "11. References",
        [
            "Dark Patterns dataset from Kaggle",
            "Synthetic dark-pattern dataset used for external validation and additional training examples",
            "Python libraries: pandas, scikit-learn, openpyxl, and reportlab documentation",
        ],
    )

    doc.build(story)
    print(f"Report created at: {REPORT_PATH}")


if __name__ == "__main__":
    create_report()
