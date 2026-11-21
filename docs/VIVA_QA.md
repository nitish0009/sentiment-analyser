# Viva Questions and Answers

**Project:** Sentiment Analysis System (binary movie-review classification)  
**Model:** Logistic Regression with TF-IDF features  
**Dataset:** IMDb 50K (49,582 rows after duplicate removal)

These are the questions most likely to be asked in a project viva, with short,
technically correct answers. Numbers come from `models/model_metadata.json`.

---

## Quick numbers to remember

| Item | Value |
|---|---|
| Raw reviews | 50,000 |
| Duplicates removed | 418 |
| Clean reviews | 49,582 |
| Training rows | 39,665 |
| Test rows | 9,917 |
| TF-IDF features | 50,000 (unigrams + bigrams) |
| Naive Bayes accuracy / F1 | 87.79% / 0.8779 |
| Logistic Regression accuracy / F1 | 89.65% / 0.8965 |
| LR confusion matrix | [[4368, 572], [454, 4523]] |
| NB confusion matrix | [[4299, 641], [570, 4407]] |
| Low-confidence threshold | 60% |

---

## Q1. What is sentiment analysis?

**Answer:** Sentiment analysis is the task of automatically identifying the
opinion or emotion expressed in text. In this project it is a supervised binary
classification problem: a review is mapped to **positive** or **negative** based
on patterns learned from labelled examples.

---

## Q2. Why did you choose the IMDb 50K dataset?

**Answer:** It is a large, public, human-labelled dataset with 25,000 positive and
25,000 negative movie reviews. The classes are balanced, the reviews are
realistic, and the dataset is a standard benchmark, so the results are easy to
compare with other work. It is also small enough to train on a laptop.

---

## Q3. Why is your model binary? Why not add a Neutral class?

**Answer:** The IMDb dataset contains only positive and negative labels. Training a
third class would require neutral examples and a separately evaluated model.
Adding a “neutral” output without such training data would be fabricated. The app
therefore reports only the two supported classes and clearly flags low-confidence
predictions instead of pretending they are neutral.

---

## Q4. What is the size of your dataset after cleaning?

**Answer:** 50,000 reviews were loaded. 418 exact duplicate review texts were
removed, leaving **49,582** reviews: 24,884 positive and 24,698 negative. The
stratified split produced **39,665** training reviews and **9,917** test reviews.

---

## Q5. Why did you use an 80/20 split?

**Answer:** 80% gives the model enough data to learn vocabulary and weights, while
20% leaves a large, representative test set (9,917 reviews). A stratified split
preserves the positive/negative ratio in both parts, and `random_state=42` makes
the split reproducible.

---

## Q6. What does `random_state=42` do?

**Answer:** It seeds the random number generator used by `train_test_split` and
Logistic Regression. With the same data and library versions, the same rows go
into training and test, so the experiment can be reproduced. It does not make the
model itself “random” in the sense of unstable predictions.

---

## Q7. Explain your preprocessing pipeline.

**Answer:**

1. Normalise apostrophes and lowercase.
2. Expand contractions (`wasn't` → `was not`).
3. Remove HTML tags/entities, URLs and e-mail addresses.
4. Remove non-letter characters and collapse whitespace.
5. Tokenise with NLTK `word_tokenize`.
6. Remove NLTK English stop-words, except a sentiment keep-list.
7. Apply Porter stemming.
8. Join the tokens back into one cleaned string for TF-IDF.

The exact same function is used in training and prediction.

---

## Q8. Why did you preserve negations if they are stop-words?

**Answer:** Removing `not` would make “good” and “not good” look almost identical
to the vectorizer, which reverses meaning. The project subtracts negations such
as `not`, `no`, `never`, `cannot` and `without` from NLTK's stop-word list.
Contrast words (`but`, `however`) and intensifiers (`very`, `really`) are also
kept because they affect sentiment.

---

## Q9. What is the difference between stemming and lemmatisation?

**Answer:** Stemming crudely chops word endings to a common root, often producing
non-words (`beautiful` → `beauti`, `studies` → `studi`). Lemmatisation uses a
dictionary and part-of-speech information to produce a valid word (`better` →
`good`). Stemming is faster; lemmatisation is more linguistically correct. This
project uses the NLTK **Porter stemmer** because it is fast and sufficient for
TF-IDF features.

---

## Q10. What is TF-IDF?

**Answer:** TF-IDF means Term Frequency–Inverse Document Frequency. A term gets a
high weight when it is frequent in a particular document but rare across the
corpus. This down-weights common words such as `the` and up-weights informative
words such as `excellent` or `boring`. The project uses sublinear TF
(`1 + log(tf)`) and L2-normalised vectors.

---

## Q11. Why did you fit TF-IDF only on the training data?

**Answer:** To prevent **data leakage**. If the vectorizer sees the test set while
building its vocabulary or IDF weights, the test data has influenced the model.
The correct sequence is:

```python
X_train = vectorizer.fit_transform(X_train_clean)
X_test  = vectorizer.transform(X_test_clean)
```

`fit` learns the vocabulary and IDF; `transform` only applies them.

---

## Q12. Why unigrams and bigrams?

**Answer:** Unigrams capture individual sentiment words (`great`, `awful`).
Bigrams capture short phrases whose meaning is not just the sum of the words
(`not good`, `waste time`, `one best`, `high recommend`). The vectorizer uses
`ngram_range=(1, 2)`.

---

## Q13. Why 50,000 features, `min_df=2` and `max_df=0.95`?

**Answer:** 50,000 is a practical cap that keeps the matrix sparse and the model
fast. `min_df=2` removes one-off typos and names that cannot generalise.
`max_df=0.95` removes tokens that appear in almost every review and therefore
carry little discriminative information. These are common, defensible defaults,
not magic numbers.

---

## Q14. What is `sublinear_tf=True`?

**Answer:** It replaces raw term frequency with `1 + log(tf)`. Repeating a word
ten times should not give it ten times the influence of one occurrence. The log
compression reduces the dominance of very long or repetitive reviews.

---

## Q15. How does Multinomial Naive Bayes work?

**Answer:** It applies Bayes' theorem and assumes features are conditionally
independent given the class. For each class it estimates the probability of every
token, then multiplies those probabilities for the tokens in a new review. It is
extremely fast and works well with sparse word features. `alpha=1.0` is Laplace
smoothing, which avoids zero probabilities for unseen words.

---

## Q16. How does Logistic Regression work?

**Answer:** It learns one weight per TF-IDF feature and computes a linear score
`z = w·x + b`. A sigmoid converts `z` into a probability between 0 and 1. The
weights are optimised by minimising log loss with L2 regularisation. Unlike
Naive Bayes, the weights are learned jointly, so overlapping words can be balanced
more effectively.

---

## Q17. Why did Logistic Regression beat Naive Bayes?

**Answer:** Logistic Regression reached 89.65% accuracy and 0.8965 weighted F1,
versus 87.79% and 0.8779 for Naive Bayes. It makes 1,026 test errors compared with
1,211. Naive Bayes assumes conditional independence, which is false for language;
Logistic Regression can learn a better-weighted linear boundary and handles
overlapping vocabulary more effectively. Naive Bayes remains a very fast
baseline.

---

## Q18. What is a confusion matrix? Give your values.

**Answer:** A confusion matrix shows true labels against predicted labels. For
Logistic Regression:

|  | Predicted Negative | Predicted Positive |
|---|---:|---:|
| True Negative | 4,368 | 572 |
| True Positive | 454 | 4,523 |

- TN = 4,368: negative reviews correctly classified.
- FP = 572: negative reviews wrongly called positive.
- FN = 454: positive reviews wrongly called negative.
- TP = 4,523: positive reviews correctly classified.

Total errors = 572 + 454 = 1,026.

---

## Q19. Accuracy vs precision vs recall vs F1?

**Answer:**

- **Accuracy** = all correct predictions / all predictions.
- **Precision** = of the reviews predicted positive, how many really are positive.
- **Recall** = of the truly positive reviews, how many were found.
- **F1** = harmonic mean of precision and recall.

Accuracy can hide class-specific errors. Precision and recall separate the two
error types, and F1 balances them.

---

## Q20. What is the difference between macro and weighted averages?

**Answer:** Macro averages give every class equal weight. Weighted averages weight
each class by its number of examples. Because this test set is almost balanced
(4,940 negative vs 4,977 positive), the two are almost identical: 0.8965 for
both in Logistic Regression.

---

## Q21. How did you select the final model?

**Answer:** The rule is: highest **weighted F1-score** on the held-out test set,
with accuracy as a tie-breaker. Logistic Regression won with F1 = 0.8965. F1 is
preferred over accuracy because it balances precision and recall across both
classes.

---

## Q22. Did you use the test set for model selection? Is that a problem?

**Answer:** Yes, the test set was used to compare the two models. This is a minor
limitation: the selected model's test score is slightly optimistic because the
same data influenced the choice. No hyperparameter grid search was performed, so
the effect is small, but a stricter design would use a separate validation split
for selection and keep the test set untouched until the final report.

---

## Q23. What files do you save with Joblib?

**Answer:**

- `models/sentiment_model.pkl` — the trained classifier;
- `models/tfidf_vectorizer.pkl` — the fitted vocabulary and IDF weights;
- `models/model_metadata.json` — metrics, split, preprocessing settings,
  top features, timings and library versions.

The metadata is plain JSON so it can be inspected without loading Python objects.

---

## Q24. How does the app avoid retraining?

**Answer:** `SentimentPredictor` loads the three saved artifacts from disk. The
app calls `transform` on the input text and `predict_proba` on the classifier. It
never calls `fit`. Training happens only when the developer runs
`python src/train.py`.

---

## Q25. What does the confidence score mean?

**Answer:** It is the model's predicted probability for the winning class, for
example 0.8965 means the model assigns about 89.65% probability to that class.
It is **not** a guarantee and is not calibrated. Logistic Regression on sparse
TF-IDF features can be overconfident on short or unusual text. The UI flags
scores below 60% as low-confidence.

---

## Q26. What happens when confidence is low?

**Answer:** The result card shows the binary prediction and the probability bars,
and a warning appears: the model is unsure between positive and negative, and it
still does not predict “neutral” because neutral labels were not in training.

---

## Q27. How do you handle empty or invalid input?

**Answer:** `SentimentPredictor.predict` rejects non-strings, blank strings and
text that becomes empty after preprocessing (for example, only punctuation). The
app shows a clear message instead of crashing. Tests cover all three cases.

---

## Q28. What tests did you write?

**Answer:** 17 tests in total. Ten cover preprocessing: lowercasing/stemming,
HTML/URL/email removal, stop-words, negation preservation, contractions,
whitespace, empty input, non-string input and special characters. Seven cover
saved artifacts and prediction: metadata, positive/negative predictions,
probability validity, empty/punctuation input, non-string input and missing
artifacts. All pass.

---

## Q29. What is data leakage, and where could it occur here?

**Answer:** Data leakage is when information from the test set influences
training. It could occur if TF-IDF were fitted on the full dataset, if duplicate
reviews appeared in both train and test, or if the app used different
preprocessing from training. The project addresses these by fitting TF-IDF on
the training split only, removing exact duplicates before splitting, and sharing
one preprocessing function.

---

## Q30. Why didn't you use BERT or an LLM?

**Answer:** The project brief is a classical NLP/ML system that a student can
explain and run on a laptop. BERT would add large dependencies, longer training,
GPU pressure and a much harder viva explanation. Classical TF-IDF with Logistic
Regression already reaches about 90% accuracy on IMDb. A transformer comparison
is listed as future work.

---

## Q31. What is the biggest weakness of your model?

**Answer:** It is a bag-of-words model, so it does not truly understand syntax,
negation scope, sarcasm or mixed opinions. For example, “Not bad at all” is
predicted negative because `bad` has a strong negative coefficient; “I did not
enjoy this movie” can be predicted positive because `enjoy` is a strong positive
token. The confidence warning helps, but it does not solve the underlying
limitation.

---

## Q32. How would you improve the project?

**Answer:** Use a separate validation split, run cross-validation, calibrate
probabilities, add negation-scope features, try character n-grams, compare a
Linear SVM, and optionally compare a small transformer. I would also add batch
prediction and per-token explanations.

---

## Q33. How can someone reproduce your results?

**Answer:**

```bash
python scripts/download_dataset.py
python src/train.py
python -m unittest discover -s tests -v
python -m streamlit run streamlit_app.py
```

The split, random state, TF-IDF settings and versions are recorded in
`models/model_metadata.json`.

---

## Q34. What was the most important engineering lesson?

**Answer:** Training and serving must use the same preprocessing and the same
fitted vectorizer. A model is only reproducible if the entire pipeline — not just
the classifier — is saved and reused. The shared `preprocess_text` function and
the saved Joblib artifacts enforce that.

---

## Q35. Why does the app store the model in `models/` instead of training at startup?

**Answer:** Training takes about a minute and would block every user request. More
importantly, retraining at startup would make results non-deterministic and waste
resources. Saving the fitted artifacts makes the app fast and ensures every user
sees the same evaluated model.

---

## Q36. If you had to explain the project in 30 seconds, what would you say?

**Answer:** “I built a binary sentiment classifier for movie reviews. I cleaned
and tokenised 49,582 IMDb reviews, converted them into 50,000 TF-IDF unigram and
bigram features fitted only on the training split, and compared Naive Bayes with
Logistic Regression. Logistic Regression won with 89.65% test accuracy and a
0.8965 weighted F1. I saved the model and vectorizer with Joblib and built a
Streamlit app that loads them, shows probabilities and flags low-confidence
predictions. It is honest about being binary and about the limits of a
bag-of-words model.”
