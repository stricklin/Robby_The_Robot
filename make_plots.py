import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_rewards = np.load("training_rewards.npy")
    testing_rewards = np.load("testing_rewards.npy")
    plt.title("Training rewards")
    plt.plot(list(range(1, len(training_rewards) * 100 + 1, 100)), training_rewards, 'blue', label="Training data")
    plt.legend(loc='lower right')
    plt.ylabel("Total rewards")
    plt.xlabel("Episodes")
    plt.show()
    plt.title("Testing rewards")
    plt.plot(list(range(1, len(testing_rewards) * 100 + 1, 100)), testing_rewards, 'red', label="Testing data")
    plt.legend(loc='lower right')
    plt.ylabel("Total rewards")
    plt.xlabel("Episodes")
    plt.show()


