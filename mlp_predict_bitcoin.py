import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

def read_data_from_csv(csv_path):
  data = pd.read_csv(csv_path, delimiter=',', header=None)
  return data

def normalize(array):
  result = (array - array.min()) / (array.max() - array.min())
  return result

def classify_with_mlpregressor(hidden_layer_sizes, file_path):
  data = read_data_from_csv(file_path)
  X, y = data.iloc[:, 1:], data.iloc[:, 0:1]
  X = normalize(X)
  y = normalize(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

  clf = MLPRegressor(
    solver='sgd',
    learning_rate='adaptive',
    activation='relu',
    momentum=0.9,
    random_state=1,
    hidden_layer_sizes=(hidden_layer_sizes,))
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  score = clf.score(X_test, y_test) * 100

  return round(score, 3)

def plot_graph(scores, days_windows):
  plt.plot(days_windows, scores)
  plt.title("Bitcoins Full Length")
  plt.xlabel("Score")
  plt.ylabel("Days Windows")
  plt.show()


def main():
  days_windows = [1, 5, 10, 20, 50, 100, 500]
  n = 5

  results_file = open("assets/results_full.txt", "w")

  while n <= 100:
    scores = []
    print("Number of neurons = " + str(n))
    results_file.write("==========================================\n")
    results_file.write("Number of neurons: " + str(n) + "\n")

    for i in days_windows:
      file_path = "assets/bitcoin_full_history/bitcoin_" + str(i) + ".csv"
      score = classify_with_mlpregressor(n, file_path)
      scores.append(score)
      print(str(i) + ": " + str(score) + "%")
      results_file.write("Days Windows: " + str(i) + " :: Score: " + str(score) + "%" + "\n")

    plot_graph(scores, days_windows)

    n += 5


main()

