import os
import re
from shutil import copyfile
import numpy as np


class DatasetSplitFactory:
    def __init__(self,
                 path_source,
                 path_train,
                 path_validation,
                 path_test,
                 id_pattern,
                 test_size=10000,
                 validation_size=10000):
        self.path_source = path_source
        self.path_train = path_train
        self.path_validation = path_validation
        self.path_test = path_test
        self.test_size = test_size
        self.validation_size = validation_size
        self.id_pattern = id_pattern

    @staticmethod
    def create_filepath(folder, id, suffix):
        filename = "{}.{}".format(id, suffix)
        return os.path.join(folder, filename)

    @staticmethod
    def create_folder_if_not_exists(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def copy_samples(self, ids, path_from, path_to):
        self.create_folder_if_not_exists(path_to)
        print("Copying {} images from {} to {}".format(len(ids), path_from, path_to))
        for id in ids:
            file_jpg = self.create_filepath(path_from, id, 'jpg')
            file_json = self.create_filepath(path_from, id, 'json')
            copyfile(file_jpg, self.create_filepath(path_to, id, 'jpg'))
            copyfile(file_json, self.create_filepath(path_to, id, 'json'))
        print("Copied {} images".format(len(ids)))

    def create_and_save(self, ids, size, from_path, to_path):
        print("Generating {} samples...".format(size))
        samples = np.random.choice(list(set(ids)), size=size,
                                        replace=False)
        self.copy_samples(samples, from_path, to_path)
        return samples

    def read_ids(self, path):
        files = os.listdir(path)
        ids = []
        for f in files:
            match = re.findall(self.id_pattern, f)
            if match:
                ids.append(match[0])
        # n_duplicates = len(ids) - len(set(ids))
        ids = list(set(ids))
        print("Processing {} ids".format(len(ids)))
        return ids

    def run(self):
        # Get all ids from the source folder (without suffix or _clean)
        ids = self.read_ids(self.path_source)

        # Create and save test data
        test_ids = self.create_and_save(ids, self.test_size, self.path_source,
                                           self.path_test)

        # Create and save validation data after excluding test ids
        unused_ids = [id for id in ids if id not in test_ids]
        validation_ids = self.create_and_save(unused_ids, self.validation_size,
                                                 self.path_source, self.path_validation)

        # Create and save training data after excluding both test and
        # validation ids
        unused_ids = [id for id in unused_ids if
                      id not in validation_ids and id not in test_ids]
        self.copy_samples(unused_ids, self.path_source, self.path_train)


def run_refined_m2u():
    factory = DatasetSplitFactory(
        path_source="../data/refined_MPII2Unity/",
        path_train="../data/refined_MPII2Unity_Train/",
        path_validation="../data/refined_MPII2Unity_Val/",
        path_test="../data/refined_MPII2Unity_Test/",
        test_size=5000,
        validation_size=5000,
        id_pattern=r'(p\d\d_\d+).jpg'
    )
    factory.run()


def run_refined_u2m():
    factory = DatasetSplitFactory(
        path_source="../data/refined_Unity2MPII/",
        path_train="../data/refined_Unity2MPII_Train/",
        path_validation="../data/refined_Unity2MPII_Val/",
        path_test="../data/refined_Unity2MPII_Test/",
        test_size=10000,
        validation_size=10000,
        id_pattern=r'(\d+).jpg'
    )
    factory.run()


def run_unityeyes():
    factory = DatasetSplitFactory(
        path_source="../data/UnityEyes/",
        path_train="../data/UnityEyesTrain/",
        path_validation="../data/UnityEyesVal/",
        path_test="../data/UnityEyesTest/",
        test_size=10000,
        validation_size=10000,
        id_pattern=r'(\d+)\.jpg'
    )
    factory.run()



if __name__ == "__main__":
    # run_unityeyes()
    # run_refined_u2m()
    run_refined_m2u()

