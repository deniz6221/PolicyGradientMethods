import numpy as np
import matplotlib.pyplot as plt
import json


f_path = "checkpoints/rewards.json"
data = json.load(open(f_path, "r"))
data = np.array(data)

window_size = 1000
moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')

lossX = [i for i in range(len(moving_avg))]

plt.plot(lossX, moving_avg)
plt.title("Reward over episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.show()