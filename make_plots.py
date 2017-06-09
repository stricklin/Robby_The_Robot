import numpy as np
import os
import matplotlib.pyplot as plt

def make_plot(name):
    rewards = np.load(name)
    name = name.rstrip(".npy")
    plt.title(name + " rewards")
    plt.plot(range(len(rewards)), rewards, 'blue', label="")
    plt.ylabel("Total rewards")
    plt.xlabel("Episodes")
    plt.savefig(name + ".png")

if __name__ == "__main__":
    files = os.listdir()
    files.sort()
    plots = []
    for file in files:
        if file.endswith(".npy"):
            plots.append(file)
    for plot in plots:
        make_plot(plot)
