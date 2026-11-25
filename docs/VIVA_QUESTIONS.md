# Viva questions and answers

(The same questions are in `Project_Report.docx`.)

## Basics
**1. What is spam detection?** A binary classification task: label a message as spam (unwanted or fraudulent) or ham (genuine).

**2. Why is it supervised learning?** Every training message has a known label, and the model learns the mapping from text to label.

**3. What dataset did you use and how big is it?** The SMS Spam Collection: 5,572 messages, 5,169 after removing 403 duplicates (4,516 ham, 653 spam).

**4. Why remove duplicates?** Identical messages can land in both training and test sets and inflate the scores.

## Text processing
**5. What is TF-IDF?** TF is how often a word occurs in a message; IDF down-weights words that occur in many messages. TF x IDF gives high scores to words that are distinctive for a message, such as "prize".

**6. TF-IDF vs Bag of Words?** Bag of Words only counts; TF-IDF also reduces the weight of common words.

**7. What are stop words and bigrams?** Stop words are very common words (the, is, and) that carry little meaning. Bigrams are pairs of consecutive words, like "claim prize".

**8. Why replace numbers, links and money signs with tokens instead of deleting them?** Every phone number or link is different, but having one is a spam signal. Tokens (`numtoken`, `urltoken`, `currencytoken`) let the model learn that pattern. Those tokens are in fact among the top spam indicators (see `top_spam_words.png`).

## Models
**9. Explain Multinomial Naive Bayes.** It uses Bayes theorem to compute the probability of each class given the words, assuming words are independent ("naive"). It is fast and strong on text.

**10. Explain Logistic Regression.** A weighted sum of word features passed through the sigmoid function gives P(spam). Above 0.5 means spam.

**11. Explain SVM.** It finds the separating hyperplane with the largest margin between the classes. A linear SVM suits sparse, high-dimensional TF-IDF data.

**12. Why is the SVM wrapped in a calibrator?** LinearSVC outputs scores, not probabilities. `CalibratedClassifierCV` converts the scores into probabilities so all models can report a spam probability and ROC/PR curves.

**13. What does class_weight="balanced" do?** It gives more weight to the rarer class (spam) so the model does not ignore it.

## Evaluation
**14. Why is accuracy not enough?** About 87% of messages are ham, so predicting "ham" for everything gives about 87% accuracy and catches no spam.

**15. Define precision, recall, F1.** Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R).

**16. What is a confusion matrix?** A table of actual vs predicted classes: TN, FP, FN, TP.

**17. What are ROC-AUC and PR-AUC?** They measure how well the model ranks spam above ham across all thresholds; PR-AUC is more informative for imbalanced data.

**18. Which error is worse?** A false positive hides a genuine message; a false negative lets spam through. Many systems favour high precision; the threshold can be adjusted.

**19. Which model won and why?** By cross-validated F1, tuned Naive Bayes (0.9526), with test F1 0.9457 and accuracy 98.65%. The gap to SVM (0.9468) and Logistic Regression (0.9400) is within about one fold standard deviation, so all three are comparable. SVM and Logistic Regression have slightly higher ROC-AUC.

**20. Why did the result differ from a simple, untuned run?** Tuning (n-grams, alpha, C), better cleaning and removing duplicates all change the scores. Small datasets and one split make rankings between close models unstable.

## Methodology
**21. What is cross-validation and why use it?** The training data is split into 5 folds; each fold is used once for validation. It gives a more stable estimate and is used to tune settings without touching the test set.

**22. What is GridSearchCV?** It tries every combination of the listed hyper-parameters, scores each with cross-validation, and keeps the best.

**23. Why did you pick the model with cross-validation and not test score?** Choosing by test score would leak information from the test set into model selection. The test set is used once at the end.

**24. What is data leakage and how did you avoid it?** Information from the test data influencing training. The split happens before TF-IDF is fitted; TF-IDF lives inside the Pipeline so it is fitted only on training folds.

**25. Why a Pipeline?** The same cleaning, TF-IDF and classifier are applied to training data, test data and new messages, and everything is saved in one file.

**26. What does stratify do?** It keeps the spam percentage the same in the training and test sets.

## Engineering
**27. How did you test the project?** 22 pytest unit tests for cleaning, data loading, model training, prediction and the CLI; ruff for style; GitHub Actions runs them on Python 3.10 to 3.12 for every push.

**28. What is CI?** Continuous integration: an automatic check that runs the tests whenever code is pushed.

**29. Limitations and future work?** Old English-only dataset, text-only features, one split. Future: larger and Indian/Hinglish datasets, nested cross-validation, threshold tuning, URL and sender features, transformer models.

## Generalisation and the unseen-message test
**30. Why did you test on extra messages if you already had a test set?** The original test set is drawn from the same old dataset as the training data, so it is optimistic. I wrote 60 new messages (modern Indian-style scams, Hinglish, genuine OTP and bank alerts) to see how the model behaves on data that differs from the training data.

**31. What were the results and what do they mean?** Accuracy dropped to 76.7% (Naive Bayes), 86.7% (Logistic Regression) and 81.7% (SVM). This is called domain shift: the model learned the vocabulary of older SMS spam and misses new styles such as fake KYC, job and investment offers. It shows that 98% on a test split does not guarantee real-world performance.

**32. Naive Bayes was chosen but Logistic Regression did better on the unseen set. Why not switch?** The model was chosen by cross-validation on the training data. Switching because of the unseen results would use the unseen set for selection, so it would no longer be an unbiased test. The correct fix is better, more recent training data. Also, 60 messages is a small sample, so the difference is not conclusive.

**33. Why were genuine OTP and bank alerts flagged as spam?** They contain numbers, amounts and urgent wording ("do not share", "debited"), which are common in spam, and such automatic messages were rare in the training data.

**34. What does the confidence level in the interface mean?** It is how far the spam probability is from 50%: above 90% High, 75 to 90% Medium, below 75% Low.

**35. What does the spam threshold slider do?** A message is called spam if its probability is at least the threshold. Lowering it catches more spam but flags more genuine messages (higher recall, lower precision); raising it does the opposite.

**36. How does the word explanation work?** For Naive Bayes and Logistic Regression, each word's TF-IDF value is multiplied by the model's weight for that word. Positive values push towards spam and negative values towards not spam.
