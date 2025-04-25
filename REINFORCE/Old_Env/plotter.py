import matplotlib.pyplot as plt
import numpy as np

import json

rps_first = json.load(open("checkpoints/rews_new_2997.json", "r"))
rps_first = np.array(rps_first)

file_name = "rews_cst_7202"
rps_lst = json.load(open(f"checkpoints/{file_name}.json", "r"))


rps_data = np.array(rps_lst)
rps_data = np.concatenate((rps_first, rps_data))


window_size = 1000


smoothed_loss = np.convolve(rps_data, np.ones(window_size)/window_size, mode='valid')
lossX = [i for i in range(len(smoothed_loss))]

plt.plot(lossX, smoothed_loss)
plt.title("Reward over episodes")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.legend()
plt.show()