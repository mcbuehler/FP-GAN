import os

from util.enum_classes import Mode
from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
from util.enum_classes import EnvironmentVariable as EV


class DatasetManager:
    def __init__(self):
        self.datasets = {
            Mode.TRAIN_UNITY: UnityDataset,
            Mode.VALIDATION_UNITY: UnityDataset,
            Mode.VALIDATION_MPII: MPIIDataset,

            Mode.INFERENCE_UNITY_TO_MPII: UnityDataset,
            Mode.INFERENCE_MPII_TO_UNITY: MPIIDataset
        }

    def get_dataset_path(self, mode):
        path = None
        if mode == Mode.TRAIN_UNITY:
            path = EV.get_value(EV.PATH_UNITY_TRAIN)
        elif mode == Mode.VALIDATION_UNITY:
            path = EV.get_value(EV.PATH_UNITY_VALIDATION)
        elif mode == Mode.VALIDATION_MPII:
            path= EV.get_value(EV.PATH_MPII)

        elif mode == Mode.INFERENCE_UNITY_TO_MPII:
            path = EV.get_value(EV.PATH_UNITY_TRAIN)
        elif mode == Mode.INFERENCE_MPII_TO_UNITY:
            path = EV.get_value(EV.PATH_MPII)

        if path is None:
            print("No dataset path found for mode '{}'".format(mode))
            print("Have you set your environment variables correctly?")

        return path

    def get_dataset_iterator_for_path(self, path, image_size, batch_size, shuffle=True, repeat=True, testing=False):
        if os.path.isdir(path):
            # We assume it is a UnityEyes dataset
            dataset = UnityDataset
        else:
            # We assume it is a hdf5 file
            dataset = MPIIDataset

        iterator = dataset(path, image_size, batch_size, shuffle=shuffle, testing=testing, repeat=repeat).get_iterator()
        return iterator