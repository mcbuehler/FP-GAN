import logging


class BaseDataset:
    """
    Base Dataset.

    Call get_iterator() to get an iterator for this dataset.
    """
    # This will be set when creating the iterator.
    N = None

    def __init__(self, path_input, image_size=(72, 120), batch_size=32, shuffle=True, buffer_size=1000, testing=False, repeat=True):
        self.path_input = path_input
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.testing = testing
        self.repeat = repeat

    def _prepare_iterator(self, dataset):
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(self.batch_size)
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

