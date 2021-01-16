from collections import defaultdict

import numpy as np
import pandas as pd

df = pd.read_csv("p5.csv")
print(f"{len(df.columns)} columns, ", end="")

df = df.replace("?", np.nan)
df = df.dropna(axis=1)
print(f"after dropping NA, {len(df.columns)} columns")

target = "class"
classes = df[target].unique()
features = df.columns[df.columns != target]


class NaiveBayes:
    def __init__(self):
        self.p_values = defaultdict(lambda: defaultdict(dict))
        self.p_classes = dict()

    def train(self, df):
        for cls in df[target].unique():
            cls_rows = df[df[target] == cls]
            self.p_classes[cls] = len(cls_rows) / len(df)

            for feature in df.columns:
                for value, count in cls_rows[feature].value_counts().iteritems():
                    self.p_values[cls][feature][value] = count / len(cls_rows)

    def classify(self, df):
        max_p = 0
        pred = None

        for clas in classes:
            p = self.p_classes[clas]
            for feature, value in df.iteritems():
                try:
                    p *= self.p_values[clas][feature][value]
                except KeyError:
                    p = 0

            if p > max_p:
                max_p = p
                pred = clas

        return pred

    def evaluate(self, df):
        arr = []
        for i in df.index:
            k = self.classify(df.loc[i, features]) == df.loc[i, target]
            arr.append(k)
        return sum(arr) / len(df)


test_df = df.sample(frac=0.3)
train_df = df.drop(test_df.index)
print(f"train: {len(train_df)}, test: {len(test_df)}")

b = NaiveBayes()
b.train(train_df)
print("train accuracy:", b.evaluate(train_df))
print("test accuracy:", b.evaluate(test_df))
