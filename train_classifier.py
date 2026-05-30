import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('emails_cleaned.csv')
print(len(df))

# Clean missing sender/text data by treating missing parts as empty strings
# This prevents NaN from propagating into the combined email text.
df['Sender'] = df['Sender'].fillna('')
df['Text'] = df['Text'].fillna('')

df['emails'] = df['Sender'] + ' ' + df['Text']
emails = df['emails'].values
og_labels = df['Label'].values

le = LabelEncoder()
labels = le.fit_transform(og_labels).ravel()

# print(emails.iloc[0])
# print(senders.iloc[0])
# print(labels[:5]) # numPy array

# split data into training and test sets
X_train_raw, X_test_raw, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

y_train = y_train.ravel()
y_test = y_test.ravel()

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# define classifiers
lr = LogisticRegression()
nb = MultinomialNB()
svm = SVC(probability=True)

# train classifiers
lr.fit(X_train, y_train)
nb.fit(X_train, y_train)
svm.fit(X_train, y_train)

#evaluate classifiers on test set
lr_score = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
nb_score = roc_auc_score(y_test, nb.predict_proba(X_test)[:, 1])
svm_score = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

# Plot ROC-AUC curve for all models
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr.predict_proba(X_test)[:, 1])
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(X_test)[:, 1])
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm.predict_proba(X_test)[:, 1])
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %0.2f)' % lr_score)
plt.plot(fpr_nb, tpr_nb, label='Naive Bayes (AUC = %0.2f)' % nb_score)
plt.plot(fpr_svm, tpr_svm, label='SVM (AUC = %0.2f)' % svm_score)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

# Print out confusion matrix and classification report for each model
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Evaluate classifiers on test set
models = [lr, nb, svm]
for i, model in enumerate(models):
    y_pred = model.predict(X_test)
    print(f"Confusion matrix for {model.__class__.__name__}:")
    print(confusion_matrix(y_test, y_pred))
    print(f"Classification report for {model.__class__.__name__}:")
    print(classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
    plt.title(f"Confusion Matrix for {model.__class__.__name__}")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.show()


# evaluation before threshold tuning (precision-recall curve)
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# get probabilities for positive class (junk)
lr_probs = lr.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, lr_probs)

plt.figure()
plt.plot(recalls, precisions)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Logistic Regression')
plt.show()

import pandas as pd
pr_table = pd.DataFrame({
    'threshold': thresholds,
    'precision': precisions[:-1],
    'recall': recalls[:-1]
})
print(pr_table[pr_table['threshold'].between(0.3, 0.9)].iloc[::10])

# evaluate at custom threshold
threshold = 0.884695
lr_probs = lr.predict_proba(X_test)[:, 1]
y_pred_custom = (lr_probs >= threshold).astype(int)

print("--- Custom Threshold Evaluation ---")
print(classification_report(y_test, y_pred_custom, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred_custom))

# cross validate
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline_lr = Pipeline([('vectorizer', CountVectorizer()), ('model', LogisticRegression(max_iter=1000))])
pipeline_nb = Pipeline([('vectorizer', CountVectorizer()), ('model', MultinomialNB())])
pipeline_svm = Pipeline([('vectorizer', CountVectorizer()), ('model', SVC(probability=True))])

for pipeline, name in zip([pipeline_lr, pipeline_nb, pipeline_svm], ['Logistic Regression', 'Naive Bayes', 'SVM']):
    scores = cross_val_score(pipeline, emails, labels, cv=cv, scoring='roc_auc')
    print(f"{name}: {scores} | Mean: {scores.mean():.3f}")

# just to see which emails it's getting wrong
y_pred_svm = svm.predict(X_test)

# Add results back to a dataframe
results = pd.DataFrame({
    'email': X_test_raw,         # original raw text (you kept this from the split)
    'true_label': le.inverse_transform(y_test),
    'predicted': le.inverse_transform(y_pred_svm)
})

# Filter to only wrong ones
wrong = results[results['true_label'] != results['predicted']]
print(f"Misclassified: {len(wrong)} out of {len(results)}")
print(wrong.head(10))

# writing to file on disk so i don't need to retrain every time
import joblib
joblib.dump(lr, 'email_classifier.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# loading back:
# import joblib
# lr = joblib.load('email_classifier.pkl')
# vectorizer = joblib.load('vectorizer.pkl')