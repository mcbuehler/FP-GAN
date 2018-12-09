import os

from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset


class DatasetManager:

    @staticmethod
    def get_dataset_iterator_for_path(path, image_size, batch_size, shuffle=True, repeat=True, testing=False, drop_remainder=False):
        if os.path.isdir(path):
            # We assume it is a UnityEyes dataset
            dataset = UnityDataset
        else:
            # We assume it is a hdf5 file
            dataset = MPIIDataset
        dataset = dataset(path, image_size, batch_size, shuffle=shuffle, testing=testing,
                repeat=repeat, drop_remainder=drop_remainder)
        iterator = dataset.get_iterator()
        iterator.N = dataset.N
        return iterator