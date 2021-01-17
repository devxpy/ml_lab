import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("p6.csv", names=["message", "label"])

X = df.message
y = df.label.map({"pos": 1, "neg": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y)
print(f"Train: {len(y_train)}, Test: {len(y_test)}, Total: {len(y)}")

count_vec = CountVectorizer()
X_train_dm = count_vec.fit_transform(X_train)
X_test_dm = count_vec.transform(X_test)

print(
    "Document-term matrix:",
    pd.DataFrame(X_train_dm.toarray(), columns=count_vec.get_feature_names()),
    sep="\n",
)
print()

classifier = MultinomialNB()
classifier.fit(X_train_dm, y_train)
y_pred = classifier.predict(X_test_dm)

print("Predictions:")
for x, pred in zip(X_train, y_pred):
    print(x, "->", "pos" if pred == 1 else "neg")
print()

print("Accuracy Metrics:")
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
