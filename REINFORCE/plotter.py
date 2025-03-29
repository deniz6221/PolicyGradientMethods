import matplotlib.pyplot as plt
import numpy as np

import json

file_name = "rewjson_4995"
rps_lst = json.load(open(f"checkpoints/{file_name}.json", "r"))


rps_data = np.array(rps_lst)


window_size = 100


smoothed_loss = np.convolve(rps_data, np.ones(window_size)/window_size, mode='valid')
lossX = [i for i in range(len(smoothed_loss))]

plt.plot(lossX, smoothed_loss)
plt.title("RPS over episodes")
plt.xlabel("Episode")
plt.ylabel("RPS")
plt.grid()
plt.legend()
plt.show()