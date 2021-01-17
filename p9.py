from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()
print("features:", iris_dataset.target_names)

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset["data"], iris_dataset["target"], random_state=0
)

print(
    f"train: {len(y_train)}, test: {len(y_train)}, total: {len(iris_dataset['data'])}"
)
knn = KNeighborsClassifier(1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print()
print("Actual, Predicted")
for y, yp in zip(y_test, y_pred):
    print(iris_dataset.target_names[y] + ", " + iris_dataset.target_names[yp])
print()

print(f"test accuracy:", knn.score(X_test, y_test))
