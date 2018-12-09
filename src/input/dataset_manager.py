import os

from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
from input.refineddataset import RefinedDataset
import logging

class DatasetManager:
    dataset_classes = {
        'unity': UnityDataset,
        'mpii': MPIIDataset,
        'refined': RefinedDataset
    }

    @classmethod
    def _get_dataset_class(cls, dataset_class):
        if dataset_class not in cls.dataset_classes.keys():
            logging.warning("Dataset class not found: {}".format(dataset_class))
            return None
        return cls.dataset_classes[dataset_class]

    @classmethod
    def get_dataset_iterator_for_path(
            cls,
            path,
            image_size,
            batch_size,
            shuffle=True,
            repeat=True,
            testing=False,
            drop_remainder=False,
            dataset_class=None
    ):
        dataset = cls._get_dataset_class(dataset_class)
        dataset = dataset(path, image_size, batch_size, shuffle=shuffle, testing=testing,
                repeat=repeat, drop_remainder=drop_remainder)
        iterator = dataset.get_iterator()
        iterator.N = dataset.N
        return iterator