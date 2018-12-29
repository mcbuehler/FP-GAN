
from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    limit = 10
    dataset = UnityDataset( "../data/UnityEyes", batch_size=1000, image_size=(72, 120), shuffle=True)
    # dataset = MPIIDataset("../data/MPIIFaceGaze/single-eye-right_zhang.h5", batch_size=1000, image_size=(72, 120), shuffle=True)
    iterator = dataset.get_iterator()
    gaze_list = list()

    with tf.Session() as sess:
        n_batches = int(dataset.N / dataset.batch_size)
        for i in range(limit):
            print(i, "/", n_batches)
            next_element = iterator.get_next()

            elem = sess.run(next_element['gaze'])
            gaze_list += list(elem)

    print(gaze_list)
    pitch, yaw = zip(*gaze_list)
    n, bins, patches = plt.hist(pitch, 50, density=True, facecolor='g', alpha=0.75)
    plt.show()
    n, bins, patches = plt.hist(yaw, 50, density=True, facecolor='g', alpha=0.75)
    plt.show()