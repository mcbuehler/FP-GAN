import json
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.enum_classes import Mode


class BaseTest:
    """
    Base class for obtaining an evaluation score for eye gaze estimation.
    Inherit from this class to get accuracy / error estimates for eye gaze
    predition models.
    """

    def __init__(self, model, mode, path, image_size, batch_size,
                 dataset_class,
                 rgb, normalise_gaze, filter_gaze):
        """

        Args:
            model: prediction model
            mode: validation, test, val_mpii,...
            path: path to input dataset
            image_size: (height, width)
            batch_size:
            dataset_class: unity, mpii or refined
            rgb: color or gray-scale images
            normalise_gaze: whether to normalise gaze
            filter_gaze: whether to filter gaze
        """
        self.iterator = DatasetManager.get_dataset_iterator_for_path(
            path,
            image_size,
            batch_size,
            shuffle=False,
            repeat=True,
            do_augmentation=False,
            dataset_class=dataset_class,
            rgb=rgb,
            normalise_gaze=normalise_gaze,
            filter_gaze=filter_gaze)
        self.path = path
        self.mode = mode
        self.n_batches_per_epoch = int(self.iterator.N / batch_size) + 1
        self.model = model
        self.outputs, self.loss = self.get_loss(summary_key="epoch")
        self.summary_op = tf.summary.merge_all(key="epoch")

    def get_loss(self, summary_key):
        """
        Obtain loss terms for given summary key
        Args:
            summary_key: string
        Returns: outputs, loss

        """
        outputs, loss = self.model.get_loss(
            self.iterator, is_training=False, mode=self.mode,
            summary_key=summary_key)
        return outputs, loss

    def _log_result(self, loss_mean, loss_std, error_angular, step):
        """
        Prints results to logging output.
        Args:
            loss_mean: mean loss score
            loss_std: std of loss scores
            error_angular: mean angular error
            step: number of steps trained

        Returns:
        """
        logging.info('-----------Test Step %d:-------------' % step)
        logging.info("Test path: {}".format(self.path))
        logging.info('  Time: {}'.format(
            datetime.now().strftime('%b-%d-%I%M%p-%G')))
        logging.info(
            '  loss {}     : {} (std: {:.4f}, angular: {:.4f})'.format(
                self.mode, loss_mean, loss_std, error_angular))


class Test(BaseTest):
    """
    Test evaluation
    """

    def run(self, sess, step, n_batches=-1, write_folder=None):
        """
        Runs a test step
        Args:
            sess: tensorflow Session
            step: number of steps trained
            n_batches: how many batches to use for obtaining test score.
                If -1 we will use the entire dataset
            write_folder: Write test results to json file (for error analysis)

        Returns:
        """
        logging.info("Preparing Testing...")
        # Don't perform on full dataset every time (too time-consuming)
        n_batches = n_batches if n_batches > 0 else self.n_batches_per_epoch

        logging.info("Running {} batches...".format(n_batches))
        results = [sess.run(
            [self.outputs,
             self.loss])
            for i in range(n_batches)]

        # loss_values is a list
        # [[angular, mse, summary], [angular, mse, summary],...]
        angular_values = [r[0]['error_angular'] for r in results]
        loss_values = [r[1] for r in results]

        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values)
        angular_error = np.mean(angular_values)

        self._log_result(loss_mean, loss_std, angular_error, step)

        if write_folder is not None:
            # n_batches, batch_size, 2
            gaze_input = [r[0]['gaze_input'][i] for r in results for i in
                          range(len(r))]
            gaze_output = [r[0]['gaze_output'][i] for r in results for i in
                           range(len(r))]
            # json.dump does not write C data types. We need built-in data types.
            to_write = {'loss': self._to_float(loss_values),
                        'angular_error': self._to_float(angular_values),
                        'gaze_input': [self._to_float(e) for e in gaze_input],
                        'gaze_output': [self._to_float(e) for e in
                                        gaze_output],
                        'test_path': self.path}
            filepath = os.path.join(write_folder, "{}_test.json".format(
                datetime.now().strftime("%Y%m%d-%H%M")))
            with open(filepath, 'w') as f:
                json.dump(to_write, f)
            logging.info("Written to {}".format(filepath))

    def _to_float(self, values):
        return [float(v) for v in values]


class ValidationTest(BaseTest):
    """
    Validation step
    """

    def run(self, sess, step, train_writer, n_batches=-1):
        """
        Runs a validation step
        Args:
            sess: tensorflow Session
            step: number of steps trained
            train_writer: tensorflow writer for summaries
            n_batches: how many batches to use for obtaining test score.
                If -1 we will use the entire dataset

        Returns:
        """
        logging.info("Preparing validation...")
        # Don't perform on full dataset every time (too time-consuming)
        n_batches = n_batches if n_batches > 0 else self.n_batches_per_epoch

        logging.info("Running {} batches...".format(n_batches))
        results = [sess.run(
            [self.outputs['error_angular'],
             self.loss, self.summary_op])
            for i in range(n_batches)]

        # loss_values is a list
        # [[angular, mse, summary], [angular, mse, summary],...]
        angular_values = [r[0] for r in results]
        loss_values = [r[1] for r in results]
        summaries = [r[2] for r in results]

        for summary in summaries:
            train_writer.add_summary(summary, step)
        train_writer.flush()

        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values)
        angular_error = np.mean(angular_values)

        self._log_result(loss_mean, loss_std, angular_error, step)


def get_validations(gazenet, path_validation_within, dataset_class_train,
                    path_validation_unity, dataset_class_validation_unity,
                    path_validation_mpii, dataset_class_validation_mpii,
                    image_size, batch_size, rgb, normalise_gaze, filter_gaze):
    """
    Helper method for creating a number of validation steps
    Args:
        gazenet: model
        path_validation_within: path to within validation set
        dataset_class_train: dataset class for training (same as within validation)
        path_validation_unity:  path to unity validation set
        dataset_class_validation_unity: unity
        path_validation_mpii: path to mpii validation set
        dataset_class_validation_mpii: mpii
        image_size: (height, width)
        batch_size:
        rgb: color or gray-scale images
        normalise_gaze: whether to normalise gaze
        filter_gaze: whether to filter gaze

    Returns: list of validation steps
    """
    all_validations = list()
    if path_validation_within is not None:
        all_validations.append(ValidationTest(
            gazenet,
            Mode.VALIDATION_WITHIN,
            path_validation_within,
            image_size,
            batch_size,
            dataset_class_train,
            rgb=rgb,
            normalise_gaze=normalise_gaze,
            filter_gaze=filter_gaze
        )
        )
    if path_validation_unity is not None:
        all_validations.append(ValidationTest(
            gazenet,
            Mode.VALIDATION_UNITY,
            path_validation_unity,
            image_size,
            batch_size,
            dataset_class_validation_unity,
            rgb=rgb,
            normalise_gaze=normalise_gaze,
            filter_gaze=filter_gaze
        ))
    if path_validation_mpii is not None:
        all_validations.append(ValidationTest(
            gazenet,
            Mode.VALIDATION_MPII,
            path_validation_mpii,
            image_size,
            batch_size,
            dataset_class_validation_mpii,
            rgb=rgb,
            normalise_gaze=normalise_gaze,
            filter_gaze=filter_gaze
        ))
    return all_validations
