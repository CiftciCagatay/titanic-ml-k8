import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

MODEL_FILE_PATH = "app/model.pkl"

# Get data ready
df = pd.read_csv('train.csv')
include = ['Age', 'Survived']
df_ = df[include]
df_ = df_.dropna()

# Train model
x = df_.iloc[:, :-1].values
y = df_.iloc[:, 1].values

model = LinearRegression()
model.fit(x, y)

# Pickle Dump
pickle.dump(model, open(MODEL_FILE_PATH, 'wb'))
