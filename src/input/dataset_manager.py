import os

from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
from input.refineddataset import RefinedDataset
import logging
from util.enum_classes import DatasetClass as DS


class DatasetManager:
    dataset_classes = {
        DS.UNITY: UnityDataset,
        DS.MPII: MPIIDataset,
        DS.REFINED: RefinedDataset
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
            do_augmentation=False,
            drop_remainder=False,
            dataset_class=None,
            filter_gaze=False,
    ):
        dataset = cls._get_dataset_class(dataset_class)
        dataset = dataset(path, image_size, batch_size, shuffle=shuffle, do_augmentation=do_augmentation,
                repeat=repeat, drop_remainder=drop_remainder, filter_gaze=filter_gaze)
        iterator = dataset.get_iterator()
        iterator.N = dataset.N
        return iterator