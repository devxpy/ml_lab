import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

# Read Cleveland Heart disease data
df = pd.read_csv("p7.csv")
df = df.replace("?", np.nan)

# Display the data
print("Few examples from the dataset are given below")
print(df.head())
print("\nAttributes and datatypes")
print(df.dtypes)

# Model Bayesian Network
model = BayesianModel(
    [
        ("age", "trestbps"),
        ("age", "fbs"),
        ("sex", "trestbps"),
        ("exang", "trestbps"),
        ("trestbps", "heartdisease"),
        ("fbs", "heartdisease"),
        ("heartdisease", "restecg"),
        ("heartdisease", "thalach"),
        ("heartdisease", "chol"),
    ]
)
# Learning CPDs using Maximum Likelihood Estimators
print("\nLearning CPDs using Maximum Likelihood Estimators...")
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Deducing with Bayesian Network
print("\nInference with Bayesian Network:")
infer = VariableElimination(model)

print("\n1.Probability of HeartDisease given age=20")
q = infer.query(variables=["heartdisease"], evidence={"age": 20})
print(q.values)

print("\n2. Probability of HeartDisease given female, chol=100")
q = infer.query(variables=["heartdisease"], evidence={"sex": 0, "chol": 100})
print(q.values)
