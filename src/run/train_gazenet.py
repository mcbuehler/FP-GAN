import logging
import os
import numpy as np
import shutil

from input.dataset_manager import DatasetManager
from util.config_loader import Config
import tensorflow as tf
from datetime import datetime

from models.gazenet import GazeNet
from util.log import write_parameter_summaries
from util.enum_classes import Mode


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')


def get_loss(iterator, gazenet, mode, regulariser=None, is_training=True):
    gaze_dict_validation, loss_validation = gazenet.get_loss(
        iterator, is_training=is_training, mode=mode, regulariser=regulariser)
    return loss_validation


class Validation:
    def __init__(self, mode, path, image_size, batch_size, dataset_class):
        self.iterator = DatasetManager.get_dataset_iterator_for_path(
            path,
            image_size,
            batch_size,
            shuffle=False,
            repeat=True,
            testing=True,
            dataset_class=dataset_class)
        self.mode = mode
        self.n_batches_per_epoch = int(self.iterator.N / batch_size) + 1

    def get_loss(self, model):
        outputs, loss_validation = model.get_loss(
            self.iterator, is_training=False, mode=self.mode)
        return outputs, loss_validation

    def _log_result(self, loss_mean, loss_std, error_angular, step, train_writer):
        logging.info('-----------Validation at Step %d:-------------' % step)
        logging.info('  Time: {}'.format(
            datetime.now().strftime('%b-%d-%I%M%p-%G')))
        logging.info(
            '  loss {}     : {} (std: {:.4f}, angular: {:.4f})'.format(self.mode, loss_mean,
                                                      loss_std, error_angular))

        summary = tf.Summary()
        summary.value.add(tag="{}/gaze_mse".format(self.mode),
                          simple_value=loss_mean)
        summary.value.add(tag="{}/angular_error".format(self.mode),
                          simple_value=error_angular)
        train_writer.add_summary(summary, step)
        train_writer.flush()

    def perform_validation_step(self, sess, model, step, train_writer):
        logging.info("Preparing validation...")
        outputs, loss = self.get_loss(model)
        logging.info("Running {} batches...".format(self.n_batches_per_epoch))
        results = [sess.run([outputs['error_angular'], loss]) for i in range(self.n_batches_per_epoch)]
        # loss_values is a list [[angular, mse], [angular, mse],...]
        loss_values = [r[1] for r in results]
        angular_values = [r[0] for r in results]

        loss_mean = np.mean(loss_values)
        loss_std = np.std(loss_values)

        angular_error = np.mean(angular_values)

        self._log_result(loss_mean, loss_std, angular_error, step, train_writer)


def train():
    # Load the config variables
    cfg = Config(FLAGS.config, FLAGS.section)
    # Variables used for both directions
    batch_size = cfg.get('batch_size_inference')
    image_size = [cfg.get('image_height'),
                  cfg.get('image_width')]
    checkpoints_dir = cfg.get('checkpoint_folder')
    model_name = cfg.get('model_name')
    norm = cfg.get('norm')
    learning_rate = cfg.get('learning_rate')
    beta1 = cfg.get('beta1')
    beta2 = cfg.get('beta2')
    path_train = cfg.get('path_train')
    n_steps = cfg.get('n_steps')
    path_validation_unity = cfg.get('path_validation_unity')
    path_validation_mpii = cfg.get('path_validation_mpii')
    dataset_class_train = cfg.get('dataset_class_train')
    dataset_class_validation_unity = cfg.get('dataset_class_validation_unity')
    dataset_class_validation_mpii = cfg.get('dataset_class_validation_mpii')
    # Indicates whether we are loading an existing model
    # or train a new one. Will be set to true below if we load an existing model.
    load_model = checkpoints_dir != ""

    checkpoint_dir_name = "checkpoints_gazenet"
    if not load_model:
        # We are not loading a saved model
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "../"+checkpoint_dir_name+"/{}_{}".format(current_time, model_name)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass

    graph = tf.Graph()

    with tf.Session(graph=graph) as sess:
        with graph.as_default():
            gazenet = GazeNet(
                batch_size=batch_size,
                image_size=image_size,
                norm=norm,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                name=model_name
            )

            # Prepare training
            if cfg.get('use_regulariser'):
                logging.info("Using a l2 regulariser")
                regulariser = tf.contrib.layers.l2_regularizer(scale=cfg.get('regularisation_lambda'))
            else:
                regulariser = None
            train_iterator = DatasetManager.get_dataset_iterator_for_path(
                path_train, image_size, batch_size,
                shuffle=True, repeat=True, testing=False,
                dataset_class=dataset_class_train
            )
            loss_train = get_loss(train_iterator, gazenet, mode=Mode.TRAIN_UNITY, regulariser=regulariser, is_training=True)
            optimizers = gazenet.optimize(loss_train)

            # Validation graphs
            validation_unity = Validation(
                Mode.VALIDATION_UNITY,
                path_validation_unity,
                image_size,
                batch_size,
                dataset_class_validation_unity
            )
            validation_mpii = Validation(
                Mode.VALIDATION_MPII,
                path_validation_mpii,
                image_size,
                batch_size,
                dataset_class_validation_mpii
            )

            # Summaries and saver
            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
            saver = tf.train.Saver()

        if load_model:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            # We copy config file
            shutil.copyfile(FLAGS.config, os.path.join(checkpoints_dir, "{}_config.init".format(model_name)))
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()

        try:
            while step < n_steps and not coord.should_stop():
                # Perform a training step
                _, loss_value, summary = (
                    sess.run(
                        [optimizers, loss_train, summary_op],
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % 100 == 0:
                    logging.info('Step {} -->  Time: {}    loss:  {}'.format(
                        step,
                        datetime.now().strftime('%b-%d-%I%M%p-%G'),
                        loss_value)
                    )

                if step > 0 and step % 1000 == 0:
                    validation_unity.perform_validation_step(sess,
                                                             gazenet,
                                                             step,
                                                             train_writer)
                    validation_mpii.perform_validation_step(sess,
                                                             gazenet,
                                                             step,
                                                             train_writer)

                    save_path = saver.save(sess,
                                           checkpoints_dir + "/model.ckpt",
                                           global_step=step)
                    logging.info("Model saved in file: %s" % save_path)
                step += 1
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt",
                                   global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            # coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
