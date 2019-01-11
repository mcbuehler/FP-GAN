
from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    limit = 90
    dataset_unity = UnityDataset(path_input= "../data/UnityEyes", batch_size=100, image_size=(36,60), shuffle=True, do_augmentation=False)
    dataset_mpii = MPIIDataset(path_input="../data/MPIIFaceGaze/single-eye-right_zhang.h5", batch_size=100, image_size=(36,60), shuffle=True)
    iterator_unity = dataset_unity.get_iterator()
    iterator_mpii = dataset_mpii.get_iterator()

    gaze_list_unity = list()
    gaze_list_mpii = list()

    with tf.Session() as sess:
        next_u = iterator_unity.get_next()
        next_m = iterator_mpii.get_next()
        for i in range(limit):
            elem_u, elem_m = sess.run([next_u['gaze'], next_m['gaze']])
            gaze_list_unity += list(elem_u)
            gaze_list_mpii += list(elem_m)

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))

    pitch_u, yaw_u = zip(*gaze_list_unity)
    pitch_m, yaw_m = zip(*gaze_list_mpii)

    # axes[0].set_title("Pitch")
    # axes[1].set_title("Yaw")

    # unnormalised unity
    axes[0, 0].hist(pitch_u, 50, density=True, facecolor='g', alpha=0.75)
    axes[0, 1].hist(yaw_u, 50, density=True, facecolor='g', alpha=0.75)

    # unnormalised mpii
    axes[1, 0].hist(pitch_m, 50, density=True, facecolor='g', alpha=0.75)
    axes[1, 1].hist(yaw_m, 50, density=True, facecolor='g', alpha=0.75)

    # Normalised eye gaze
    axes[2, 0].hist(np.array(pitch_m)/np.pi, 50, density=True, facecolor='g', alpha=0.75)
    axes[2, 1].hist(np.array(yaw_m)/np.pi, 50, density=True, facecolor='g', alpha=0.75)

    plt.xlim([-np.pi, np.pi])
    #plt.show()

    plt.savefig('../visualisations/gaze_distributions.png')
