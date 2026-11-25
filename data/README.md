# Dataset

`raw/spam.csv` is the **SMS Spam Collection** (5,572 English SMS messages labelled `ham` or `spam`).

| Column | Meaning |
|---|---|
| `v1` | label: `ham` (genuine) or `spam` |
| `v2` | the message text |
| 3 more columns | almost empty, ignored by the code |

- Source: UCI Machine Learning Repository (Almeida & Hidalgo), also distributed on Kaggle.
- Licence: Creative Commons Attribution 4.0 (CC BY 4.0).
- After removing 403 duplicate messages: 5,169 messages (4,516 ham, 653 spam).

Citation: Almeida, T. A., Gómez Hidalgo, J. M., Yamakami, A. (2011). *Contributions to the study of SMS spam
filtering: new collection and results.* Proceedings of the 2011 ACM Symposium on Document Engineering.

To refresh the file from the original source run `python scripts/download_data.py`.
