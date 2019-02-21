import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from models.gazenet import GazeNetInference
import os

path = "../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw/20190120-1357_test.json"

path = "../checkpoints_gazenet/20190118-1514_gazenet_u_augmented_bw/20190119-1400_test.json"

path = "../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw/20190122-1639_test.json"

with open(path, 'r') as f:
    data = json.load(f)



num_bins = 10
# x = data['angular_error']
# n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
# plt.show()
# print(min(x), max(x))


pitch_true = [p[0] for p in data['gaze_input']]
pitch_pred = [p[0] for p in data['gaze_output']]
yaw_true = [p[1] for p in data['gaze_input']]
yaw_pred = [p[1] for p in data['gaze_output']]

print(len(pitch_pred), len(pitch_true))
print(len(yaw_pred), len(yaw_true))

plt.hist(pitch_true, bins=num_bins, alpha=0.5, label='y')
plt.hist(pitch_pred, bins=num_bins, alpha=0.5, label='y_hat')
plt.legend(loc='upper right')
plt.title("Error Distribution: Pitch")
# pyplot.legend(loc='upper right')
plt.show()
plt.savefig(os.path.join('../visualisations/', "pitch.png"))
plt.close()

plt.hist(yaw_true, bins=num_bins, alpha=0.5, label='y')
plt.hist(yaw_pred, bins=num_bins, alpha=0.5, label='y_hat')
plt.legend(loc='upper right')
plt.show()
plt.savefig(os.path.join('../visualisations/', "yaw.png"))
plt.close()