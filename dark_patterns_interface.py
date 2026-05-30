from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import joblib


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "best_model.joblib"
CATEGORY_MODEL_PATH = BASE_DIR / "outputs" / "best_category_model.joblib"
WINDOW_TITLE = "Text Analyzer"
CATEGORY_EXPLANATIONS = {
    "Urgency": "This wording pushes fast action using countdowns, deadlines, or immediate consequences.",
    "Scarcity": "This wording creates pressure by suggesting limited stock, limited slots, or low availability.",
    "Social Proof": "This wording nudges behavior by implying that many other people are buying or viewing it.",
    "Misdirection": "This wording tries to steer attention or decision-making in a misleading way.",
    "Obstruction": "This wording makes it harder to refuse, exit, or complete a user-friendly action.",
    "Sneaking": "This wording hides or slips in details the user may miss at first glance.",
    "Forced Action": "This wording pressures the user to do something extra before they can continue.",
    "Scam / Phishing": "This wording resembles fraud or phishing prompts that ask for urgent clicks, verification, or sensitive details.",
}


def load_model(model_path: Path, *, required: bool = True):
    if not model_path.exists():
        if required:
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run dark_patterns_project.py first."
            )
        return None
    return joblib.load(model_path)


def load_models():
    binary_model = load_model(MODEL_PATH, required=True)
    category_model = load_model(CATEGORY_MODEL_PATH, required=False)
    return binary_model, category_model


def predict_text(binary_model, category_model, text: str) -> dict[str, str]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Please enter some text before checking.")

    predicted_label = int(binary_model.predict([cleaned_text])[0])

    if predicted_label == 1:
        verdict = "Likely fake / manipulative"
        category = "Likely manipulation type: checking unavailable"
        detail = (
            "This text looks similar to scam, phishing, or dark-pattern wording "
            "such as pressure, bait, or unsafe account prompts."
        )
        color = "#b42318"
        if category_model is not None:
            predicted_category = str(category_model.predict([cleaned_text])[0])
            category = f"Likely manipulation type: {predicted_category}"
            category_detail = CATEGORY_EXPLANATIONS.get(
                predicted_category,
                "This wording matches a manipulative pattern learned from the training data.",
            )
            detail = f"{detail} {category_detail}"
        else:
            detail = (
                f"{detail} The binary detector is available, but the category model "
                "has not been generated yet."
            )
    else:
        verdict = "Likely not fake / normal"
        category = "Likely manipulation type: none"
        detail = (
            "This text looks closer to ordinary, legitimate, and "
            "non-manipulative wording."
        )
        color = "#067647"

    return {
        "verdict": verdict,
        "category": category,
        "detail": detail,
        "color": color,
    }


def build_app() -> tk.Tk:
    binary_model, category_model = load_models()

    root = tk.Tk()
    root.title(WINDOW_TITLE)
    root.geometry("960x760")
    root.minsize(820, 680)
    root.configure(bg="#f5efe6")

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Primary.TButton", font=("Segoe UI Semibold", 11), padding=10)
    style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=8)

    title_label = tk.Label(
        root,
        text=WINDOW_TITLE,
        font=("Georgia", 24, "bold"),
        bg="#f5efe6",
        fg="#1f2937",
    )
    title_label.pack(pady=(24, 6))

    subtitle_label = tk.Label(
        root,
        text=(
            "Paste a sentence, ad, alert, or message below. "
            "This model is trained on dark-pattern and scam-like text, "
            "so 'fake' here means deceptive, manipulative, or phishing-style wording."
        ),
        font=("Segoe UI", 11),
        bg="#f5efe6",
        fg="#475467",
        wraplength=780,
        justify="center",
    )
    subtitle_label.pack(padx=24)

    card = tk.Frame(root, bg="#fffaf5", bd=1, relief="solid", highlightthickness=0)
    card.pack(fill="both", expand=True, padx=28, pady=24)

    input_label = tk.Label(
        card,
        text="Enter text to analyze",
        font=("Segoe UI Semibold", 12),
        bg="#fffaf5",
        fg="#1f2937",
        anchor="w",
    )
    input_label.pack(fill="x", padx=18, pady=(18, 8))

    text_box = tk.Text(
        card,
        height=10,
        wrap="word",
        font=("Segoe UI", 11),
        bd=1,
        relief="solid",
        padx=12,
        pady=12,
        bg="#ffffff",
        fg="#101828",
        insertbackground="#101828",
    )
    text_box.pack(fill="both", expand=True, padx=18)

    button_row = tk.Frame(card, bg="#fffaf5")
    button_row.pack(fill="x", padx=18, pady=16)

    result_frame = tk.Frame(card, bg="#fdf2e8", bd=1, relief="solid", height=150)
    result_frame.pack(fill="x", expand=False, padx=18, pady=(0, 18))
    result_frame.pack_propagate(False)

    verdict_var = tk.StringVar(value="Result will appear here.")
    category_var = tk.StringVar(value="Likely manipulation type will appear here.")
    detail_var = tk.StringVar(
        value=(
            "Try a suspicious message like a reward claim alert, "
            "or a normal product or account message."
        )
    )

    verdict_label = tk.Label(
        result_frame,
        textvariable=verdict_var,
        font=("Segoe UI Semibold", 15),
        bg="#fdf2e8",
        fg="#1f2937",
        anchor="w",
    )
    verdict_label.pack(fill="x", padx=14, pady=(14, 6))

    category_label = tk.Label(
        result_frame,
        textvariable=category_var,
        font=("Segoe UI", 11, "italic"),
        bg="#fdf2e8",
        fg="#475467",
        anchor="w",
    )
    category_label.pack(fill="x", padx=14, pady=(0, 6))

    detail_label = tk.Message(
        result_frame,
        textvariable=detail_var,
        font=("Segoe UI", 11),
        bg="#fdf2e8",
        fg="#475467",
        width=760,
        justify="left",
    )
    detail_label.pack(fill="x", padx=14, pady=(0, 14))

    def refresh_layout(event=None) -> None:
        detail_label.configure(width=max(result_frame.winfo_width() - 40, 500))

    result_frame.bind("<Configure>", refresh_layout)

    def set_result(result: dict[str, str]) -> None:
        verdict_var.set(result["verdict"])
        category_var.set(result["category"])
        detail_var.set(result["detail"])
        verdict_label.configure(fg=result["color"])

    def analyze_text() -> None:
        try:
            result = predict_text(binary_model, category_model, text_box.get("1.0", "end"))
        except ValueError as exc:
            messagebox.showwarning("Missing text", str(exc))
            return
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        set_result(result)

    def clear_text() -> None:
        text_box.delete("1.0", "end")
        verdict_var.set("Result will appear here.")
        category_var.set("Likely manipulation type will appear here.")
        detail_var.set(
            "Try a suspicious message like a reward claim alert, or a normal product or account message."
        )
        verdict_label.configure(fg="#1f2937")
        text_box.focus_set()

    def fill_example(text: str) -> None:
        text_box.delete("1.0", "end")
        text_box.insert("1.0", text)
        analyze_text()

    analyze_button = ttk.Button(
        button_row,
        text="Check Text",
        style="Primary.TButton",
        command=analyze_text,
    )
    analyze_button.pack(side="left")

    clear_button = ttk.Button(
        button_row,
        text="Clear",
        style="Secondary.TButton",
        command=clear_text,
    )
    clear_button.pack(side="left", padx=10)

    example_suspicious_button = ttk.Button(
        button_row,
        text="Try Suspicious Example",
        style="Secondary.TButton",
        command=lambda: fill_example(
            "2 lakh rupees credited in your account, click the below link to claim."
        ),
    )
    example_suspicious_button.pack(side="right")

    example_normal_button = ttk.Button(
        button_row,
        text="Try Normal Example",
        style="Secondary.TButton",
        command=lambda: fill_example(
            "You can return this product within 30 days if it does not meet your needs."
        ),
    )
    example_normal_button.pack(side="right", padx=10)

    text_box.focus_set()
    root.bind("<Control-Return>", lambda event: analyze_text())
    root.bind("<Escape>", lambda event: clear_text())

    return root


def main() -> None:
    try:
        app = build_app()
    except FileNotFoundError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Model missing", str(exc))
        root.destroy()
        return

    app.mainloop()


if __name__ == "__main__":
    main()
