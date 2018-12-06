import logging
import os
import numpy as np
from input.dataset_manager import DatasetManager

import tensorflow as tf
from datetime import datetime

from models.gazenet import GazeNet
from util.log import write_parameter_summaries
from util.enum_classes import Mode


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 128, 'batch size, default: 512')
tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')
tf.flags.DEFINE_string('norm', 'batch',
                       '[instance, batch] use instance norm or batch norm, default: instance')

tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')

tf.flags.DEFINE_bool('use_regulariser', False,
                      'regulariser')
tf.flags.DEFINE_float('regularisation_lambda', 1e-4,
                      'lambda for regularisation term, default: 0.1')
tf.flags.DEFINE_float('beta1', 0.9,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_string('path_train', '../data/UnityEyesTrain',
                       'folder containing UnityEyes')
tf.flags.DEFINE_string('path_validation_unity', '../data/UnityEyesVal',
                       'folder containing UnityEyes')
tf.flags.DEFINE_string('path_validation_mpii', '../data/MPIIFaceGaze/single-eye-right_zhang.h5',
                       'file containing MPIIGaze. We only use eyes from one side.')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('n_steps', 100000,
                        'number of steps to train. Half of the steps will be trained with a fix learning rate, the second half with linearly decaying LR.')
tf.flags.DEFINE_string('data_format', 'NHWC',
                       'NHWC or NCHW. default: NHWC')  # Important: This implementation does not yet support NCHW, so stick to NHWC!


def get_loss(iterator, gazenet, mode, regulariser=None, is_training=True):
    gaze_dict_validation, loss_validation = gazenet.get_loss(
        iterator, is_training=is_training, mode=mode, regulariser=regulariser)
    return loss_validation


class Validation:
    def __init__(self, mode, path, image_size, batch_size):
        self.iterator = DatasetManager.get_dataset_iterator_for_path(
            path,
            image_size,
            batch_size,
            shuffle=False,
            repeat=True,
            testing=True)
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
    checkpoint_dir_name = "checkpoints_gazenet"
    if FLAGS.load_model is not None:
        checkpoints_dir = "../"+checkpoint_dir_name+"/" + FLAGS.load_model.lstrip(
            checkpoint_dir_name+"/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "../"+checkpoint_dir_name+"/{}".format(current_time)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass

    graph = tf.Graph()

    # make sure to use image_height first
    image_size = (FLAGS.image_height, FLAGS.image_width)

    with tf.Session(graph=graph) as sess:
        with graph.as_default():
            gazenet = GazeNet(
                batch_size=FLAGS.batch_size,
                image_size=image_size,
                norm=FLAGS.norm,
                learning_rate=FLAGS.learning_rate,
                beta1=FLAGS.beta1,
                beta2=FLAGS.beta2,
                tf_session=sess,
                name="gazenet"
            )

            # Prepare training
            if FLAGS.use_regulariser:
                logging.info("Using a l2 regulariser")
                regulariser = tf.contrib.layers.l2_regularizer(scale=FLAGS.regularisation_lambda)
            else:
                regulariser = None
            train_iterator = DatasetManager.get_dataset_iterator_for_path(
                FLAGS.path_train, image_size, FLAGS.batch_size,
                shuffle=True, repeat=True, testing=False
            )
            loss_train = get_loss(train_iterator, gazenet, mode=Mode.TRAIN_UNITY, regulariser=regulariser, is_training=True)
            optimizers = gazenet.optimize(loss_train)

            # Validation graphs
            validation_unity = Validation(
                Mode.VALIDATION_UNITY,
                FLAGS.path_validation_unity,
                image_size,
                FLAGS.batch_size
            )
            validation_mpii = Validation(
                Mode.VALIDATION_MPII,
                FLAGS.path_validation_mpii,
                image_size,
                FLAGS.batch_size
            )

            # Summaries and saver
            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
            saver = tf.train.Saver()

        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[2].split(".")[0])
        else:
            write_parameter_summaries(gazenet, os.path.join(checkpoints_dir, "config.json"))
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()

        try:
            while step < FLAGS.n_steps and not coord.should_stop():
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

                if step > 0 and step % 5000 == 0:
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
