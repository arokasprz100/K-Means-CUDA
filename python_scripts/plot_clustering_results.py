import pandas as pd
import seaborn as sns

data_gpu = pd.read_csv("../data/clusters_gpu.data", header=None)
data_gpu.columns = [*data_gpu.columns[:-1], "cluster"]
sns_plot = sns.pairplot(data_gpu, hue="cluster")
sns_plot.savefig("clustering_gpu.png")
print("GPU results figure saved to clustering_gpu.png file")

data_cpu = pd.read_csv("../data/clusters_cpu.data", header=None)
data_cpu.columns = [*data_cpu.columns[:-1], "cluster"]
sns_plot = sns.pairplot(data_cpu, hue="cluster")
sns_plot.savefig("clustering_cpu.png")
print("CPU results figure saved to clustering_cpu.png file")
