from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
SYNTHETIC_SHEET_NAME = "Data"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TEXT_COLUMNS = ["page_id", "text", "label", "Pattern Category"]
MODEL_SORT_COLUMNS = ["f1", "accuracy", "precision"]
CATEGORY_SORT_COLUMNS = ["macro_f1", "accuracy", "weighted_f1"]
MODEL_NAME_COLUMN = "model"
NEGATIVE_CATEGORY = "Not Dark Pattern"
POSITIVE_LABEL = 1
BEST_BINARY_MODEL_FILE = "best_model.joblib"
BEST_CATEGORY_MODEL_FILE = "best_category_model.joblib"
BINARY_RESULTS_FILE = "model_results.csv"
CATEGORY_RESULTS_FILE = "category_model_results.csv"
TFIDF_PARAMS = {
    "stop_words": "english",
    "ngram_range": (1, 2),
    "min_df": 1,
    "sublinear_tf": True,
}


@dataclass(frozen=True)
class DatasetConfig:
    source_name: str
    file_path: Path
    split_prefix: str
    excel_sheet: str | None = None


def resolve_data_file(candidates: list[Path], dataset_name: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find the {dataset_name} dataset. Checked:\n{candidate_text}"
    )


def build_dataset_configs() -> list[DatasetConfig]:
    synthetic_file = resolve_data_file(
        [
            BASE_DIR / "synthetic_dark_pattern_dataset_2500.xlsx",
            BASE_DIR / "outputs" / "synthetic_dark_pattern_dataset_2500.xlsx",
            Path(
                r"c:\Users\KrutarthPC\Downloads\synthetic_dark_pattern_dataset_2500.xlsx"
            ),
        ],
        "synthetic",
    )

    return [
        DatasetConfig(
            source_name="original",
            file_path=BASE_DIR / "dataset.tsv",
            split_prefix="original",
        ),
        DatasetConfig(
            source_name="synthetic",
            file_path=synthetic_file,
            split_prefix="synthetic",
            excel_sheet=SYNTHETIC_SHEET_NAME,
        ),
        DatasetConfig(
            source_name="scam",
            file_path=BASE_DIR / "scam_dataset.tsv",
            split_prefix="scam",
        ),
    ]


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(**TFIDF_PARAMS)


def build_category_model() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]
    )


def read_dataset_file(file_path: Path, *, excel_sheet: str | None = None) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if excel_sheet is not None or suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=excel_sheet)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    raise ValueError(f"Unsupported dataset format: {file_path}")


def load_dataset(
    file_path: Path,
    source_name: str,
    *,
    excel_sheet: str | None = None,
) -> pd.DataFrame:
    df = read_dataset_file(file_path, excel_sheet=excel_sheet)
    missing_columns = [column for column in TEXT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{file_path} is missing required columns: {', '.join(missing_columns)}"
        )

    df = df[TEXT_COLUMNS].copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["source_dataset"] = source_name
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split(df: pd.DataFrame, file_name: str) -> None:
    df.to_csv(OUTPUT_DIR / file_name, index=False)


def create_model_builders() -> dict[str, Callable[[], Pipeline]]:
    return {
        "Linear SVM": lambda: Pipeline(
            [
                ("tfidf", build_vectorizer()),
                ("classifier", LinearSVC(class_weight="balanced")),
            ]
        ),
        "Logistic Regression": lambda: Pipeline(
            [
                ("tfidf", build_vectorizer()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                ),
            ]
        ),
        "SGD Log Loss": lambda: Pipeline(
            [
                ("tfidf", build_vectorizer()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Complement NB": lambda: Pipeline(
            [
                ("tfidf", build_vectorizer()),
                ("classifier", ComplementNB()),
            ]
        ),
        "Hybrid Stacking": lambda: Pipeline(
            [
                ("tfidf", build_vectorizer()),
                (
                    "classifier",
                    StackingClassifier(
                        estimators=[
                            (
                                "lr",
                                LogisticRegression(
                                    max_iter=2000,
                                    class_weight="balanced",
                                ),
                            ),
                            (
                                "svm",
                                CalibratedClassifierCV(
                                    LinearSVC(class_weight="balanced"),
                                    cv=3,
                                ),
                            ),
                            ("cnb", ComplementNB()),
                        ],
                        final_estimator=LogisticRegression(max_iter=2000),
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, Pipeline]:
    model_builders = create_model_builders()
    results = []
    best_name = None
    best_model = None
    best_score = None

    for name, build_model in model_builders.items():
        model = build_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        metrics = {
            MODEL_NAME_COLUMN: name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        }
        results.append(metrics)

        current_score = tuple(metrics[column] for column in MODEL_SORT_COLUMNS)
        if best_score is None or current_score > best_score:
            best_score = current_score
            best_name = name
            best_model = model

    results_df = pd.DataFrame(results).sort_values(
        by=MODEL_SORT_COLUMNS,
        ascending=False,
    )
    return results_df, best_name, best_model


def build_category_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def filter_positive_rows(df: pd.DataFrame) -> pd.DataFrame:
        filtered_df = df[
            (df["label"] == POSITIVE_LABEL)
            & df["Pattern Category"].notna()
            & (df["Pattern Category"] != NEGATIVE_CATEGORY)
        ].copy()
        return filtered_df.reset_index(drop=True)

    return filter_positive_rows(train_df), filter_positive_rows(test_df)


def evaluate_category_model(
    category_train_df: pd.DataFrame,
    category_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Pipeline]:
    if category_train_df.empty or category_test_df.empty:
        raise ValueError("Not enough positive samples to train the category model.")

    category_model = build_category_model()
    category_model.fit(
        category_train_df["text"],
        category_train_df["Pattern Category"],
    )
    predictions = category_model.predict(category_test_df["text"])

    metrics = {
        MODEL_NAME_COLUMN: "Linear SVM",
        "accuracy": accuracy_score(category_test_df["Pattern Category"], predictions),
        "macro_f1": f1_score(
            category_test_df["Pattern Category"],
            predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            category_test_df["Pattern Category"],
            predictions,
            average="weighted",
            zero_division=0,
        ),
        "train_rows": len(category_train_df),
        "test_rows": len(category_test_df),
        "category_count": category_train_df["Pattern Category"].nunique(),
    }

    return pd.DataFrame([metrics]).sort_values(
        by=CATEGORY_SORT_COLUMNS,
        ascending=False,
    ), category_model


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    dataset_configs = build_dataset_configs()
    split_results = []

    for config in dataset_configs:
        dataset_df = load_dataset(
            config.file_path,
            config.source_name,
            excel_sheet=config.excel_sheet,
        )
        train_df, test_df = split_dataset(dataset_df)
        split_results.append((config, dataset_df, train_df, test_df))

        save_split(train_df, f"{config.split_prefix}_train_split.csv")
        save_split(test_df, f"{config.split_prefix}_test_split.csv")

    train_df = pd.concat(
        [train_df for _, _, train_df, _ in split_results],
        ignore_index=True,
    )
    test_df = pd.concat(
        [test_df for _, _, _, test_df in split_results],
        ignore_index=True,
    )

    save_split(train_df, "train_split.csv")
    save_split(test_df, "test_split.csv")

    results_df, best_name, best_model = evaluate_models(
        train_df["text"],
        test_df["text"],
        train_df["label"],
        test_df["label"],
    )
    category_train_df, category_test_df = build_category_splits(train_df, test_df)
    category_results_df, category_model = evaluate_category_model(
        category_train_df,
        category_test_df,
    )

    results_df.to_csv(OUTPUT_DIR / BINARY_RESULTS_FILE, index=False)
    category_results_df.to_csv(OUTPUT_DIR / CATEGORY_RESULTS_FILE, index=False)
    joblib.dump(best_model, OUTPUT_DIR / BEST_BINARY_MODEL_FILE)
    joblib.dump(category_model, OUTPUT_DIR / BEST_CATEGORY_MODEL_FILE)

    for config, dataset_df, train_split, test_split in split_results:
        print(f"{config.source_name.title()} dataset path: {config.file_path}")
        print(f"{config.source_name.title()} cleaned rows: {len(dataset_df)}")
        print(f"{config.source_name.title()} training rows: {len(train_split)}")
        print(f"{config.source_name.title()} testing rows: {len(test_split)}")
        print()

    print("Merged training rows:", len(train_df))
    print("Merged testing rows:", len(test_df))
    print("\nResults:")
    print(results_df.to_string(index=False))
    print("\nBest model:", best_name)
    print("\nCategory model results:")
    print(category_results_df.to_string(index=False))
    print(f"\nCategory model labels: {category_train_df['Pattern Category'].nunique()}")


if __name__ == "__main__":
    main()
