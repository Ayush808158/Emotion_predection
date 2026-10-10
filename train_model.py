"""Step 2 - Train the emotion model.

Run:  python train_model.py          (full search)
      python train_model.py --fast   (quick test)

What it does:
  1. Loads + cleans data/train.txt (data_cleaning.py).
  2. Uses the SAME 80/20 split as the original notebook (random_state=42),
     so the new accuracy can be compared directly with the old 86.2%.
  3. Tries 3 models (word TF-IDF + LogReg, word+char TF-IDF + LogReg,
     word+char TF-IDF + LinearSVC) and picks the best with cross-validation
     on the 80% training part only.
  4. Scores the winner ONCE on the 20% test part and saves everything.

Output: models/emotion_model.joblib, reports/results.txt, reports/confusion_matrix.png
"""
import argparse
import platform
import re
import time
from pathlib import Path

import joblib
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from data_cleaning import load_data

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "emotion_model.joblib"
REPORT_DIR = ROOT / "reports"
SEED = 42


def word_features():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=150_000)


def word_char_features():
    return FeatureUnion([
        ("word", word_features()),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                                 sublinear_tf=True, max_features=200_000)),
    ])


def candidates(fast):
    lr_c, svc_c = ([10], [0.3]) if fast else ([10, 30, 100], [0.3, 1.0, 3.0])
    return {
        "word TF-IDF + LogisticRegression": (
            Pipeline([("features", word_features()), ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": lr_c}),
        "word+char TF-IDF + LogisticRegression": (
            Pipeline([("features", word_char_features()), ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": lr_c}),
        "word+char TF-IDF + LinearSVC": (
            Pipeline([("features", word_char_features()), ("clf", LinearSVC())]),
            {"clf__C": svc_c}),
    }


def pin_sklearn_version():
    """Write the scikit-learn version into requirements.txt (model must be loaded with the same one)."""
    req = ROOT / "requirements.txt"
    if req.exists():
        pin = f"scikit-learn=={sklearn.__version__}"
        lines = [pin if re.match(r"\s*scikit-learn\b", ln) else ln for ln in req.read_text().splitlines()]
        req.write_text("\n".join(lines) + "\n")
        print(f"requirements.txt -> {pin}")


def save_confusion_matrix(y_true, y_pred, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="small search for a quick test")
    args = parser.parse_args()
    start = time.time()

    data = load_data()
    train, test = train_test_split(data, test_size=0.20, random_state=SEED)
    print(f"Rows: {len(data):,}  (train {len(train):,} / test {len(test):,})")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    leaderboard, best = [], None
    for name, (pipe, grid) in candidates(args.fast).items():
        t0 = time.time()
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1)
        search.fit(train["text"], train["emotion"])
        leaderboard.append((name, search.best_score_, search.best_params_))
        print(f"  {name:<42} CV acc {search.best_score_:.4f}  {search.best_params_}  ({time.time() - t0:.0f}s)")
        if best is None or search.best_score_ > best[0]:
            best = (search.best_score_, name, search.best_estimator_, search.best_params_)

    _, best_name, best_pipe, best_params = best
    model = clone(best_pipe)
    if isinstance(model.named_steps["clf"], LinearSVC):   # SVC has no probabilities -> calibrate
        model.set_params(clf=CalibratedClassifierCV(model.named_steps["clf"], cv=5))
    model.fit(train["text"], train["emotion"])

    pred = model.predict(test["text"])                    # test set used exactly once
    acc = accuracy_score(test["emotion"], pred)
    macro_f1 = f1_score(test["emotion"], pred, average="macro")
    report = classification_report(test["emotion"], pred, digits=3)
    print(f"\nBEST: {best_name} {best_params}")
    print(f"TEST accuracy: {acc:.4f}  |  macro-F1: {macro_f1:.4f}\n{report}")

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    (REPORT_DIR / "results.txt").write_text(
        f"Best model: {best_name} {best_params}\nTest accuracy: {acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}")
    save_confusion_matrix(test["emotion"], pred, list(model.classes_))
    meta = {"best_model": best_name, "test_accuracy": round(float(acc), 4),
            "test_macro_f1": round(float(macro_f1), 4), "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(), "n_train": len(train), "n_test": len(test)}
    joblib.dump({"pipeline": model, "meta": meta}, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB) in {time.time() - start:.0f}s")
    pin_sklearn_version()


if __name__ == "__main__":
    main()
