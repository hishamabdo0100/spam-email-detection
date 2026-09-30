# 📧 Spam Email Detection

Binary classification of emails (**spam vs. ham**) using TF-IDF features and three classical ML models —
Logistic Regression, LinearSVC and KNN — tuned with `GridSearchCV` and compared with ROC / PR curves,
confusion matrices and test-set metrics.

![Word clouds](images/word_clouds.png)

## Results (held-out test set, 1,139 emails)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| **LinearSVC (final)** | **99.56%** | 98.91% | 99.27% | 99.09% |
| Logistic Regression | 99.47% | 98.55% | 99.27% | 98.91% |
| KNN (cosine / distance-weighted) | 98.60% | 96.40% | 97.81% | 97.10% |

The final LinearSVC misclassified only **5 of 1,139** test emails (3 false positives, 2 false negatives).

![Model comparison](images/model_comparison.png)
![Confusion matrices](images/confusion_matrices.png)

## Pipeline
1. **EDA** – class balance, email length, most frequent words, word clouds
2. **Cleaning** – 33 duplicate emails removed (5,728 → 5,695), no missing values
3. **Features** – TF-IDF (word uni/bi-grams, sublinear TF, English stop-words removed)
4. **Baselines** – Logistic Regression, LinearSVC, KNN evaluated with 3-fold cross-validation
5. **Tuning** – `GridSearchCV` over n-gram range, `min_df`, `max_df` and each model's main hyper-parameter
6. **Evaluation** – ROC / Precision-Recall curves, confusion matrices, test-set metrics
7. **Interpretation** – top words pushing an email towards spam or ham, plus a `predict_email()` helper

The data is imbalanced (~76% ham / 24% spam), so `class_weight='balanced'` is used and precision / recall / F1 are reported alongside accuracy.

![ROC and PR curves](images/roc_pr_curves.png)
![Top features](images/top_features.png)

## Limitations
The strongest "ham" indicators are words like `enron`, `vince`, `kaminski` and `houston`, and the years `2000` / `2005`
also act as signals. This shows the ham emails come mostly from a single corporate mailbox, so the near-perfect score
partly reflects **dataset bias**. Performance on emails from other sources will likely be lower; a more diverse dataset
would be needed before real-world use.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook spam_email_detection.ipynb
```
The notebook expects `emails.csv` (columns: `text`, `spam`) in the same folder.

## Tech stack
Python · Pandas · NumPy · Scikit-learn · NLTK · Matplotlib · Seaborn · WordCloud
