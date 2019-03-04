import os
import re
from os import symlink
import numpy as np


# Base directory containing the checkpoint folder for trained GAN models
BASE_DIR = os.getenv('FPGAN_BASE_DIR', '/disks/data4/marcel/')


class DatasetSplitFactory:
    """
    Base class for splitting datasets into train, (val), and test set.
    """
    def __init__(self,
                 path_source,
                 path_train,
                 path_validation,
                 path_test,
                 id_pattern):
        self.path_source = path_source
        self.path_train = path_train
        self.path_validation = path_validation
        self.path_test = path_test
        self.id_pattern = id_pattern

    @staticmethod
    def create_filepath(folder, id, suffix):
        """
        Helper method to create a file path.
        Args:
            folder: path to a folder
            id: stem of filename
            suffix: e.g. ".jpg"

        Returns: path
        """
        filename = "{}.{}".format(id, suffix)
        return os.path.join(folder, filename)

    @staticmethod
    def create_folder_if_not_exists(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def symlink_samples(self, ids, path_from, path_to):
        """
        Creates symlinks for given files (ids) in path_to.
        Args:
            ids: list of file stems
            path_from: folder containing files in ids
            path_to: target folder (will be created if it does not exist)

        Returns:

        """
        self.create_folder_if_not_exists(path_to)
        print("Copying {} images from {} to {}".format(len(ids), path_from, path_to))
        for id in ids:
            file_jpg_in = self.create_filepath(path_from, id, 'jpg')
            file_json_in = self.create_filepath(path_from, id, 'json')

            file_jpg_out = self.create_filepath(path_to, id, 'jpg')
            file_json_out = self.create_filepath(path_to, id, 'json')

            if os.path.isfile(file_jpg_out) or os.path.isfile(file_json_out):
                # There exists an old version. Remove those symlinks.
                os.unlink(file_jpg_out)
                os.unlink(file_json_out)

            symlink(file_jpg_in, file_jpg_out)
            symlink(file_json_in, file_json_out)
        print("Copied {} images".format(len(ids)))

    def create_and_save(self, all_ids, size, from_path, to_path):
        """
        Creates a random subset from given ids.
        Args:
            all_ids: list of file stems
            size: size of sample to be generated
            from_path: folder containing files in all_ids
            to_path: target folder (will be created if it does not exist)

        Returns: sampled ids
        """
        print("Generating {} samples...".format(size))
        samples = np.random.choice(list(set(all_ids)), size=size,
                                   replace=False)
        self.symlink_samples(samples, from_path, to_path)
        return samples

    def read_ids(self, path):
        """
        Args:
            path: path containing files

        Returns: ids under path that match with self.id_pattern

        """
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
        """
        Use this method in order to run the splitting.
        Returns:
        """
        raise NotImplementedError("Use a subclass!")


class StandardDatasetSplitFactory(DatasetSplitFactory):
    def __init__(self, test_size=10000,
                 validation_size=10000, **kwargs):
        super().__init__(**kwargs)
        self.test_size = test_size
        self.validation_size = validation_size

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
        self.symlink_samples(unused_ids, self.path_source, self.path_train)


class RefinedMPIIDatasetSplitFactory(DatasetSplitFactory):
    """
    Splits the refined MPIIGaze dataset
    """
    def __init__(self, test_person_identifiers, **kwargs):
        """
        Args:
            test_person_identifiers: numerical id for the people that should
                be used in the test set.
            **kwargs: Arguments for base class
        """
        super().__init__(**kwargs)
        self.test_person_identifiers = test_person_identifiers

    def get_test_ids(self, all_ids, prefix_list, from_path, to_path):
        """
        We select the subset of ids that are used for testing.
        Args:
            all_ids: list of file stems
            prefix_list: Used as prefix filter
            from_path: folder containing files in all_ids
            to_path: target folder (will be created if it does not exist)

        Returns:

        """
        test_ids = [id for id in all_ids if id[:3] in prefix_list]
        self.symlink_samples(test_ids, from_path, to_path)
        return test_ids

    def run_for_person_identifiers(self, test_person_identifiers=list()):
        """
        Runs the splitting for given person identifiers
        Args:
            test_person_identifiers: list of person identifiers
                e.g. ["p01", "p02"]

        Returns:

        """
        # Get all ids from the source folder (without suffix or _clean)
        ids = self.read_ids(self.path_source)

        # Create and save test data
        test_ids = self.get_test_ids(ids, test_person_identifiers, self.path_source, self.path_test)

        # Create and save training data after excluding both test and
        # validation ids
        unused_ids = [id for id in ids if id not in test_ids]
        self.symlink_samples(unused_ids, self.path_source, self.path_train)

    def run(self):
        self.run_for_person_identifiers(self.test_person_identifiers)


class MPIIDatasetSplitFactory(DatasetSplitFactory):
    def __init__(self, test_person_identifiers, **kwargs):
        """
        Args:
            test_person_identifiers: numerical id for the people that should
                be used in the test set.
            **kwargs: Arguments for base class
        """
        super().__init__(**kwargs)
        self.test_person_identifiers = test_person_identifiers

    def run(self):
        n_train, n_test = self.run_for_person_identifiers(self.test_person_identifiers)
        print("Written train / test: {} / {}".format(n_train, n_test))

    def write_all(self, data, person_identifier, out_file):
        """
        Writes data to h5 out_file for single person_identifier.
        Args:
            data: dict with person_identifier and attributes
            person_identifier: e.g. "p01"
            out_file: opened h5 file

        Returns:

        """
        g = out_file.create_group(person_identifier)
        for attr in data[person_identifier]:
            d = data[person_identifier][attr]
            g.create_dataset(attr, data=d)
        n = data[person_identifier]['gaze'].shape[0]
        return n

    def run_for_person_identifiers(self, test_person_identifiers):
        """
        Runs the splitting for given person identifiers
        Args:
            test_person_identifiers: list of person identifiers
                e.g. ["p01", "p02"]

        Returns:

        """
        import h5py
        file_train = h5py.File(self.path_train, 'w')
        file_test = h5py.File(self.path_test, 'w')
        n_train = 0
        n_test = 0

        with h5py.File(self.path_source, 'r') as hf:
            person_identifiers = hf.keys()
            for person_identifier in person_identifiers:
                if person_identifier in test_person_identifiers:
                    n_test += self.write_all(hf, person_identifier, file_test)
                else:
                    n_train += self.write_all(hf, person_identifier, file_train)

        file_train.close()
        file_test.close()

        return n_train, n_test


def run_refined_r2s(model_identifier):
    gan_checkpoint_path = os.path.join(BASE_DIR, os.path.join("checkpoints", model_identifier))
    test_ids = ["p{}".format(i) for i in map(lambda s: str(s).zfill(2), range(12, 15))]
    factory = RefinedMPIIDatasetSplitFactory(
        path_source=os.path.join(gan_checkpoint_path, "refined_MPII2Unity/"),
        path_train=os.path.join(gan_checkpoint_path, "refined_MPII2Unity_Train/"),
        path_validation=None,
        path_test=os.path.join(gan_checkpoint_path, "refined_MPII2Unity_Test/"),
        test_person_identifiers=test_ids,
        id_pattern=r'(p\d\d_\d+).jpg'
    )
    factory.run()


def run_refined_s2r(model_identifier):
    gan_checkpoint_path = os.path.join(BASE_DIR, os.path.join("checkpoints", model_identifier))
    factory = StandardDatasetSplitFactory(
        path_source=os.path.join(gan_checkpoint_path, "refined_Unity2MPII"),
        path_train=os.path.join(gan_checkpoint_path, "refined_Unity2MPII_Train/"),
        path_validation=os.path.join(gan_checkpoint_path, "refined_Unity2MPII_Val/"),
        path_test=os.path.join(gan_checkpoint_path, "refined_Unity2MPII_Test/"),
        test_size=10000,
        validation_size=10000,
        id_pattern=r'(\d+).jpg'
    )
    factory.run()


def run_unityeyes():
    factory = StandardDatasetSplitFactory(
        path_source=os.path.join(BASE_DIR, "UnityEyesRHP"),
        path_train="../data/UnityEyesRHPTrain/",
        path_validation="../data/UnityEyesRHPVal/",
        path_test="../data/UnityEyesRHPTest/",
        test_size=10000,
        validation_size=10000,
        id_pattern=r'(\d+)\.jpg'
    )
    factory.run()


def run_mpii():
    test_ids = ["p{}".format(i) for i in map(lambda s: str(s).zfill(2), range(12, 15))]
    factory = MPIIDatasetSplitFactory(
        path_source=os.path.join(BASE_DIR, "MPIIFaceGaze/single-eye-right_zhang.h5"),
        path_train="../data/MPIIFaceGaze/train-right.h5",
        path_validation=None,
        path_test="../data/MPIIFaceGaze/test-right.h5",
        test_person_identifiers=test_ids,
        id_pattern=None
    )
    factory.run()


if __name__ == "__main__":
    # run_unityeyes()
    # run_mpii()

    checkpoint_folder = "20190118-1522_ege_l30"
    checkpoint_folder = "20190116-2305_lm_l15"
    run_refined_s2r(checkpoint_folder)
    run_refined_r2s(checkpoint_folder)

