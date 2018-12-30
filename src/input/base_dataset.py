import logging
import multiprocessing
import tensorflow as tf


class BaseDataset:
    """
    Base Dataset.

    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True, buffer_size=1000, testing=False, repeat=True, num_parallel_calls=None, drop_remainder=False, filter_gaze=False):
        self.path_input = path_input
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.testing = testing
        self.repeat = repeat
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls is not None else multiprocessing.cpu_count() - 1
        self.drop_remainder = drop_remainder
        self.filter_gaze = filter_gaze

        if self.filter_gaze:
            self.gaze_filter_range = {
                    'pitch': (-0.7, 0.2),
                    'yaw': (-0.7, 0.7)
                }

    def _is_in_gaze_range(self, gaze):
        gfr_p, gfr_y = self.gaze_filter_range['pitch'], self.gaze_filter_range['yaw']
        return tf.logical_and(
            tf.logical_and(gfr_p[0] < gaze[0], gaze[0] < gfr_p[1]),
            tf.logical_and(gfr_y[0] < gaze[1], gaze[1] < gfr_y[1])
        )

    def _prepare_iterator(self, dataset):
        if self.filter_gaze:
            dataset = dataset.filter(
                lambda s: self._is_in_gaze_range(s['gaze']))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=self.drop_remainder)
        dataset = dataset.prefetch(self.buffer_size)
        if self.repeat:
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def get_iterator(self):
        raise NotImplementedError("Implement in subclass!")

    def _iterator_ready_info(self):
        logging.info("Dataset loaded from '{}'. {} images to serve".
                     format(self.path_input, self.N))

    def get_n_batches_per_epoch(self):
        n_batches = int(self.N / self.batch_size) + 1
        return n_batches

