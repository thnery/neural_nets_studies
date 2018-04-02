import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def read_data_from_csv(csv_path):
  data = pd.read_csv(csv_path, delimiter=',', header=None)
  return data

def normalize(array):
  result = (array - array.min()) / (array.max() - array.min())
  return result

def classify_with_mlpregressor(file_path):
  data = read_data_from_csv(file_path)
  X, y = data.iloc[:, 1:], data.iloc[:, 0:1]
  X = normalize(X)
  y = normalize(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

  clf = MLPRegressor(
    solver='sgd',
    learning_rate='adaptive',
    activation='relu',
    random_state=0)
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  score = clf.score(X_test, y_test) * 100

  return round(score, 3)


def main():
  rows = [1, 5, 10, 20, 50, 100, 500]

  for i in rows:
    score = classify_with_mlpregressor("assets/bitcoin_past_300_days/bitcoin_" + str(i) + ".csv")
    print(str(i) + ": " + str(score) + "%")


main()

