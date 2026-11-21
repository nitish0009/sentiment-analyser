# Demonstration Script

A **5–7 minute** plan for presenting the Sentiment Analysis System to a teacher,
examiner or project panel. It assumes the model has already been trained and the
app starts successfully.

---

## Before the demo (5 minutes before)

1. Activate the environment:

   ```bash
   cd "sentiment analyzer"
   source .venv/bin/activate
   ```

2. Confirm that the artifacts exist:

   ```bash
   ls models/
   ```

   You should see `sentiment_model.pkl`, `tfidf_vectorizer.pkl` and
   `model_metadata.json`.

3. Start the app:

   ```bash
   python -m streamlit run streamlit_app.py
   ```

4. Open <http://127.0.0.1:8501> and wait for the sidebar badge **Model ready**.

5. Keep a terminal open for the CLI fallback:

   ```bash
   python src/predict.py "This movie was absolutely wonderful!" --show-cleaned
   ```

6. If the internet is unavailable, do not download anything during the demo. The
   dataset and artifacts are already local.

---

## Demo flow

### 0:00–0:40 — Open with the problem and scope

Show the **Home** page.

> “This is a binary sentiment analysis system for movie reviews. It uses classical
> NLP and machine learning — TF-IDF, Naive Bayes and Logistic Regression — with no
> LLMs or paid APIs. The dataset is the IMDb 50K set. I removed 418 exact
> duplicates, leaving 49,582 reviews, and used a stratified 80/20 split.”

Point to the **Model Snapshot** card:

> “The selected model is Logistic Regression with 89.65% test accuracy and a
> weighted F1 of 0.8965. Those numbers come from the held-out test set, not from
> the training data.”

---

### 0:40–2:00 — Positive and negative predictions

Click **Sentiment Analyzer** in the sidebar.

Paste or load the first positive sample:

> “I absolutely loved this film. The acting was excellent, the story kept me
> engaged from start to finish, and the ending was deeply satisfying. I would
> happily watch it again.”

Press **Analyze sentiment**. Expected result: **Positive, about 93.9%**.

> “The model reports the predicted class, the confidence, and both class
> probabilities. The text is cleaned with the same pipeline used in training:
> contractions, HTML, stop-words and Porter stemming.”

Open **See preprocessing details** briefly:

> “This is the exact token string passed to TF-IDF. Stemming turns ‘absolutely’ and
> ‘satisfying’ into stems, which is why some tokens look unusual.”

Now load the negative movie sample:

> “A complete waste of time. The plot made no sense, the acting was wooden, and I
> kept checking my watch waiting for it to end. I cannot recommend this to anyone.”

Expected result: **Negative, about 98.5%**.

> “Same pipeline, opposite class. The probability bars make the decision
> transparent.”

---

### 2:00–3:00 — Show the honest limitation

Paste:

> “I liked parts of it and disliked other parts.”

Expected result: **Positive, about 57.6%, Low confidence**, followed by a warning.

> “The model is binary. It cannot output Neutral because there are no neutral
> labels in IMDb. Instead of inventing one, it reports low confidence and warns the
> user. This is a mixed review, and a bag-of-words model is genuinely unsure.”

Optional stronger example (if asked about negation):

> “Not bad at all.”

The model predicts **Negative** with high confidence because `bad` has a strong
negative weight and the model does not explicitly understand negation scope.

> “This is a real limitation of classical bag-of-words models. Keeping `not` helps,
> but it does not solve negation scope.”

---

### 3:00–4:30 — Show the measured evaluation

Click **Model Performance**.

1. Point to the selected-model banner and headline metrics.
2. Scroll to the comparison chart:

   > “Both models use the same split, the same preprocessing and the same TF-IDF
   > matrix. Logistic Regression wins on accuracy, precision, recall and F1:
   > 89.65% versus 87.79%, and 1,026 errors versus 1,211.”

3. Scroll to the confusion matrices:

   > “For Logistic Regression, 4,368 negatives and 4,523 positives are correct.
   > There are 572 false positives and 454 false negatives. The errors are fairly
   > balanced, which is what we want.”

4. Open the **Classification reports** tabs:

   > “The weighted and macro averages are almost identical because the test classes
   > are balanced: 4,940 negative and 4,977 positive.”

5. Scroll to **What the model learned**:

   > “The strongest positive coefficients are `great`, `excel`, `enjoy`, `perfect`
   > and `love`. The strongest negative coefficients are `worst`, `bad`, `wast`,
   > `bore` and `poor`. These are global model signals, not a per-review
   > explanation.”

---

### 4:30–5:30 — Show the sample playground

Click **Sample Reviews**.

> “Six clear examples are provided, alternating positive and negative, plus three
> tricky cases: a mixed review, sarcasm, and a genuinely mixed short review.”

Press **Analyze this review** on the negative movie card.

> “The card loads the text into the analyzer and runs the saved model
> automatically. This is useful for a live demo because there is no typing.”

Optionally test the sarcasm card:

> “Oh great, another sequel that nobody asked for. Truly a masterpiece... if you
> enjoy wasting money.”

Expected result: **Negative, about 69.9%, Moderate confidence**.

> “The model picks up words like `wast`, but it does not understand sarcasm. The
> moderate confidence is appropriate.”

---

### 5:30–6:30 — Explain the engineering and close

Return to the Home page or stay on Sample Reviews.

> “Training and serving are separate. `python src/train.py` cleans the data, fits
> TF-IDF on the training split only, trains both models, evaluates them, selects
> the best weighted F1 and saves the classifier, vectorizer and metadata with
> Joblib. The Streamlit app only loads those artifacts — it never retrains.”

> “The same `preprocess_text` function is imported by training and prediction, so
> there is no train/serve mismatch. The project has 17 automated tests covering
> preprocessing and saved artifacts.”

> “Limitations: it is binary, the test set was used for model selection, confidence
> is not calibrated, and sarcasm/negation scope are hard. Future work includes a
> validation split, cross-validation, probability calibration and a small
> transformer comparison.”

> “Thank you. I’m happy to take questions.”

---

## Fallback plan

### If the app fails to start

Use the CLI:

```bash
python src/predict.py "This movie was absolutely wonderful!" --show-cleaned
python src/predict.py "A dull, badly acted mess." --show-cleaned
```

Explain the same result card verbally, then show the evaluation charts in
`reports/` and the measured metrics in `evaluation_report.md`.

### If a chart is missing

Run training again:

```bash
python src/train.py
```

The app embeds chart PNGs directly, so it does not depend on an external image
server. If the Streamlit media endpoint misbehaves, the pinned
Streamlit/Starlette versions and the base64 embedding are the workaround.

### If NLTK data is missing

```bash
python scripts/setup_nltk.py
chmod -R go-w ~/nltk_data
```

Then restart the app.

### If the model files are missing

```bash
python src/train.py
```

Training takes roughly one to two minutes.

---

## Likely questions during the demo

| Question | Short answer |
|---|---|
| Why not Neutral? | IMDb has no neutral labels; the app flags low confidence instead. |
| Why Logistic Regression? | Highest weighted F1 on the test set: 0.8965 vs 0.8779. |
| Did you leak the test set? | No: TF-IDF is fitted on train only and duplicates were removed before splitting. |
| Why keep negations? | “good” and “not good” must not collapse to the same features. |
| What is your biggest error? | Negation scope and sarcasm; see “not bad at all”. |
| How do you reproduce it? | Download dataset, run `src/train.py`; settings and versions are in the metadata. |
| Why no BERT? | Classical, explainable, laptop-friendly project; transformer comparison is future work. |

---

## One-sentence summary to memorise

> “I built a leakage-free TF-IDF and Logistic Regression sentiment classifier on
> 49,582 IMDb reviews; it reaches 89.65% test accuracy, and a Streamlit app serves
> the saved model with honest confidence reporting and no fake neutral class.”
