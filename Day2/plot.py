import matplotlib.pyplot as plt
import numpy as np
import mlflow

fig, ax = plt.subplots()
ax.plot(np.arange(10), np.random.rand(10))
plt.title("Example plot")

plt.savefig("Plot.png")

mlflow.log_artifact("Plot.png")