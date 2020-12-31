import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

number_of_clusters = 3
number_of_rows_per_cluster = 100000
centers = [(10, 10), (-5, -9), (-12, 21)]

mean = 0.5
standard_deviation = 3

data = pd.DataFrame(columns=["X", "Y"])

for cluster in range(number_of_clusters):
    center = centers[cluster]
    X = center[0] + np.random.normal(mean, standard_deviation, (number_of_rows_per_cluster, 1))
    Y = center[1] + np.random.normal(mean, standard_deviation, (number_of_rows_per_cluster, 1))
    cluster_data = pd.DataFrame(np.concatenate((X, Y), axis=1), columns=["X", "Y"])
    data = data.append(cluster_data)

data = data.reset_index(drop=True).sample(frac=1).reset_index(drop=True)

sns.pairplot(data, markers='x')
plt.show()

data.to_csv("../data/test_dataset.data", header=False, index=False)
