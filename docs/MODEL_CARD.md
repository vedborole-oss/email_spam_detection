# Model card: Email & SMS Spam Detector

**Version** 1.1.0 | **Author** YOUR_NAME | **Licence** MIT

## Intended use
Educational demonstration of text classification. Given a short message, estimate the probability that it is
spam. Not intended to be the only protection in a real mail or messaging system.

## Model
- Features: TF-IDF on cleaned text (links, currency and numbers replaced by tokens, English stop words removed).
- Candidates: Multinomial Naive Bayes, Logistic Regression (balanced class weights), linear SVM with
  probability calibration (balanced class weights).
- Tuning: `GridSearchCV`, 5-fold stratified cross-validation, scoring = F1 of the spam class.
- Selection: highest mean cross-validated F1 on the training data. Currently selected: **Naive Bayes**
  (alpha 0.1, unigrams + bigrams, min_df 1).

## Data
SMS Spam Collection, 5,572 messages; 5,169 after removing exact duplicates (4,516 ham = 87.4%,
653 spam = 12.6%). Stratified split: 4,135 training and 1,034 test messages (random_state = 42).

## Performance (test set, spam = positive class)
| Metric | Value |
|---|---|
| Accuracy | 0.9865 |
| Precision | 0.9606 |
| Recall | 0.9313 |
| F1 | 0.9457 |
| ROC AUC | 0.9925 |
| Errors | 5 genuine messages flagged as spam, 9 spam messages missed (of 1,034) |

Cross-validation F1 is 0.9526 with a standard deviation of 0.012 between folds, so differences of about
one percentage point between models are within the noise.

## Limitations and risks
- **Domain shift (measured):** the data is older English, mostly UK/Singapore SMS. On 60 hand-written,
  modern, unseen messages (Indian-style scams, Hinglish, genuine OTP and bank alerts) the selected Naive
  Bayes model reached only 76.7% accuracy and 60% spam recall (18 of 30 spam caught), versus 98.7% on the
  original test split. Fake-job and investment offers were missed most (about 20% caught). Genuine OTP and
  bank-debit alerts were flagged as spam by some models. Logistic Regression did best on this set (86.7%).
  The set is small (60 messages), so this is a rough indicator, not a precise estimate.
- **False positives** hide genuine messages; **false negatives** let scams through. The default threshold is
  0.5 and can be changed in `SpamClassifier(threshold=...)`.
- Imbalanced data (about 1 spam per 7 ham). Accuracy alone is misleading, so F1, precision and recall are
  reported.
- Only text is analysed; an attacker can rewrite messages to avoid spam words.
- Reported numbers come from one split; repeated splits or nested cross-validation would give tighter estimates.

## Unseen-message evaluation
60 hand-written messages stored in `data/unseen/unseen_messages.csv`; removed if they overlap the training
data (a test checks this) and never used for training or model selection.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes (selected) | 0.767 | 0.900 | 0.600 | 0.720 |
| Logistic Regression | 0.867 | 0.958 | 0.767 | 0.852 |
| SVM | 0.817 | 0.952 | 0.667 | 0.784 |

The model selected on the original data is not the best on shifted data. Selection was deliberately not
changed using this set, so that it remains an honest unseen test. Recommended improvement: retrain with
recent and Indian-context messages.

## Ethical considerations
Messages are personal data. Do not train on private messages without consent and do not log user input
from the web app.

## How to reproduce
```bash
pip install -e ".[app,dev]"
spam-detector train
```
