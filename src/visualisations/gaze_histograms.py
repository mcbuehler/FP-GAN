
from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.pylab as pylab
# params = {'legend.fontsize': 'xx-large',
#          'axes.labelsize': 'xx-large',
#          'axes.titlesize':'xx-large',
#          'xtick.labelsize':'xx-large',
#          'ytick.labelsize':'xx-large'}
# pylab.rcParams.update(params)


if __name__ == "__main__":
    fontsize = 40
    fontsize_ticks = 24
    limit = 370
    # limit = 2
    # dataset_unity = UnityDataset(path_input= "../data/UnityEyes", batch_size=100, image_size=(36,60), shuffle=True, do_augmentation=False)
    dataset_mpii = MPIIDataset(path_input="../data/MPIIFaceGaze/single-eye-right_zhang.h5", batch_size=100, image_size=(36,60), shuffle=True)
    # iterator_unity = dataset_unity.get_iterator()
    iterator_mpii = dataset_mpii.get_iterator()

    # gaze_list_unity = list()
    gaze_list_mpii = list()

    with tf.Session() as sess:
        # next_u = iterator_unity.get_next()
        next_m = iterator_mpii.get_next()
        for i in range(limit):
            # elem_u, elem_m = sess.run([next_u['gaze'], next_m['gaze']])
            elem_m = sess.run(next_m['gaze'])
            # gaze_list_unity += list(elem_u)
            gaze_list_mpii += list(elem_m)

        # fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 20))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    # pitch_u, yaw_u = zip(*gaze_list_unity)
    pitch_m, yaw_m = zip(*gaze_list_mpii)
    axes[0].set_ylabel('Count', fontsize=fontsize)
    # axes[0].set_title("Pitch")
    # axes[1].set_title("Yaw")
    axes[0].hist(pitch_m, density=False, facecolor='darkblue', alpha=0.75)

    axes[1].hist(yaw_m, density=False, facecolor='darkblue', alpha=0.75)

    cols = ["Pitch", "Yaw"]
    # Pitch and Yaw limits. Copied from base_dataset.py
    limits = ((-0.7, 0.2), (-0.7, 0.7))
    for ax, col, lim in zip(axes, cols, limits):
        ax.plot((lim[0], lim[1]), (1, 1), color="r", linewidth=10)
        # Column title
        ax.set_title(col, fontsize=fontsize)

    # unnormalised mpii
    # axes[1, 0].hist(pitch_m, 50, density=True, facecolor='g', alpha=0.75)
    # axes[1, 1].hist(yaw_m, 50, density=True, facecolor='g', alpha=0.75)
    #
    # # Normalised eye gaze
    # axes[2, 0].hist(np.array(pitch_m)/np.pi, 50, density=True, facecolor='g', alpha=0.75)
    # axes[2, 1].hist(np.array(yaw_m)/np.pi, 50, density=True, facecolor='g', alpha=0.75)


    # plt.xlim([-np.pi, np.pi])
    # plt.show()

    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_ticks)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize_ticks)
            # specify integer or one of preset strings, e.g.
            # tick.label.set_fontsize('x-small')

    # plt.show()
    plt.savefig('../visualisations/gaze_hist_mpii.png')
