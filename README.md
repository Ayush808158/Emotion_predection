# 🎭 Emotion Detection from Text (NLP)

Classifies English sentences into **joy, sadness, anger, fear, love or surprise** using TF-IDF features and
linear models (scikit-learn), with a Streamlit web app.

**Live demo:** _add your Streamlit link here after deploying_

## Results

Same 80/20 split (`random_state=42`) of the 16,000 labelled sentences for every row:

| Version | Approach | Test accuracy |
|---|---|---|
| v1 (`notebooks/NLP_baseline.ipynb`) | Stop-words removed, TF-IDF + Logistic Regression | 86.2% |
| **v2 (this repo)** | Word + character n-gram TF-IDF, Linear SVM (calibrated) | **89.0%** (macro-F1 85.7%) |

Model choice and `C` were selected with 3-fold cross-validation on the training part only; the test part was
scored once. Per-class results: `reports/results.txt`, confusion matrix: `reports/confusion_matrix.png`.

### What improved the score
- Stop words are **no longer removed** (NLTK's list contains "not", "no", "very", "too").
- Word 1-2-grams **plus** character 2-5-grams (robust to misspellings and word forms).
- Compared Logistic Regression and Linear SVM and tuned them with cross-validation.
- Fixed a bug in the v1 notebook (TF-IDF Naive Bayes was evaluated with the wrong model).

## Files

| File | Purpose |
|---|---|
| `data_cleaning.py` | loads `data/train.txt` and cleans the text (also used by the app) |
| `train_model.py` | trains, tunes, evaluates and saves `models/emotion_model.joblib` |
| `app.py` | Streamlit web application |
| `data/train.txt` | dataset |
| `models/emotion_model.joblib` | trained model used by the app |
| `notebooks/NLP_baseline.ipynb` | original v1 notebook |
| `requirements.txt` | dependencies |

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py              # the trained model is already included
```

To retrain (takes ~2 minutes): `python train_model.py` (`--fast` for a quick test).
It also pins your scikit-learn version in `requirements.txt`, because a saved model must be loaded with the
same version it was trained with.

## Deploy (free): Streamlit Community Cloud

1. Push this repo to GitHub.
2. Open https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repo, branch `main`, main file `app.py`, then **Deploy**.

## Limitations

- English only; trained on short, informal, first-person tweets.
- **Negation is handled poorly** (e.g. "i do not feel happy" may still be predicted as joy). The dataset labels
  come from hashtags, so negated sentences are often labelled with the emotion word itself.
- Labels are noisy and "surprise" / "love" are rare classes, which limits achievable accuracy for any model.
- Sarcasm, slang and mixed emotions are not handled.

## Data & credit

*CARER: Contextualized Affect Representations for Emotion Recognition* (Saravia et al., EMNLP 2018),
[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion). The dataset is provided for educational and
research use.
