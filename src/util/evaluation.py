import json
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf

from input.dataset_manager import DatasetManager
from util.enum_classes import Mode
import os


class BaseTest:
    def __init__(self, model, mode, path, image_size, batch_size, dataset_class,
            rgb, normalise_gaze):
        self.iterator = DatasetManager.get_dataset_iterator_for_path(
            path,
            image_size,
            batch_size,
            shuffle=False,
            repeat=True,
            do_augmentation=False,
            dataset_class=dataset_class,
            rgb=rgb,
            normalise_gaze=normalise_gaze)
        self.path = path
        self.mode = mode
        self.n_batches_per_epoch = int(self.iterator.N / batch_size) + 1
        self.model = model
        self.outputs, self.loss = self.get_loss(summary_key="epoch")
        self.summary_op = tf.summary.merge_all(key="epoch")

    def get_loss(self, summary_key):
        outputs, loss_validation = self.model.get_loss(
            self.iterator, is_training=False, mode=self.mode, summary_key=summary_key)
        return outputs, loss_validation

    def _log_result(self, loss_mean, loss_std, error_angular, step):
        logging.info('-----------Test Step %d:-------------' % step)
        logging.info("Test path: {}".format(self.path))
        logging.info('  Time: {}'.format(
            datetime.now().strftime('%b-%d-%I%M%p-%G')))
        logging.info(
            '  loss {}     : {} (std: {:.4f}, angular: {:.4f})'.format(
                self.mode, loss_mean, loss_std, error_angular))
        # summary = tf.Summary()
        # summary.value.add(tag="{}/gaze_mse".format(self.mode),
        #                   simple_value=loss_mean)
        # summary.value.add(tag="{}/angular_error".format(self.mode),
        #                   simple_value=error_angular)
        #
        # train_writer.add_summary(summary, step)
        # train_writer.flush()


class Test(BaseTest):

    def run(self, sess, step, n_batches=-1, write_folder=None):
        logging.info("Preparing Testing...")
        # Don't perform on full dataset every time (too time-consuming)
        n_batches = n_batches if n_batches > 0 else self.n_batches_per_epoch

        logging.info("Running {} batches...".format(n_batches))
        results = [sess.run(
            [self.outputs['error_angular'],
             self.loss])
            for i in range(n_batches)]

        # loss_values is a list
        # [[angular, mse, summary], [angular, mse, summary],...]
        angular_values = [r[0] for r in results]
        loss_values = [r[1] for r in results]

        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values)
        angular_error = np.mean(angular_values)

        self._log_result(loss_mean, loss_std, angular_error, step)

        if write_folder is not None:
            # json.dump does not write C data types. We need built-in data types.
            to_write = {'loss': [float(l) for l in loss_values],
                        'angular_error': [float(l) for l in angular_values],
                        'test_path': self.path }
            filepath = os.path.join(write_folder, "{}_test.json".format(datetime.now().strftime("%Y%m%d-%H%M")))
            with open(filepath, 'w') as f:
                json.dump(to_write, f)


class ValidationTest(BaseTest):

    def run(self, sess, step, train_writer, n_batches=-1):
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


def get_validations(gazenet, path_validation_within, dataset_class_train, path_validation_unity, dataset_class_validation_unity, path_validation_mpii, dataset_class_validation_mpii, image_size, batch_size, rgb, normalise_gaze):
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
            normalise_gaze=normalise_gaze
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
            normalise_gaze=normalise_gaze
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
            normalise_gaze=normalise_gaze
    ))
    return all_validations
