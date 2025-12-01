from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

ds = load_dataset("sms_spam")
clf = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
                ("svm", LinearSVC())])
clf.fit(ds["train"]["sms"], ds["train"]["label"])
pred = clf.predict(ds["test"]["sms"])
print(classification_report(ds["test"]["label"], pred, target_names=["ham","spam"]))
joblib.dump(clf, "spam_clf.joblib")
print("Saved spam_clf.joblib")
