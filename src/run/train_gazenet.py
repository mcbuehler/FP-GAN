import logging
import os

import tensorflow as tf
from datetime import datetime

from models.gazenet import GazeNet
from input.unitydataset import UnityDataset
from input.mpiidataset import MPIIDataset
from util.log import write_parameter_summaries
from util.enum_classes import Mode


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 128, 'batch size, default: 512')
tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')
tf.flags.DEFINE_string('norm', 'batch',
                       '[instance, batch] use instance norm or batch norm, default: instance')

tf.flags.DEFINE_float('learning_rate', 0.0002,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('regularisation_lambda', 0.01,
                      'lambda for regulariation term, default: 0.1')
tf.flags.DEFINE_float('beta1', 0.9,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_string('path_train', '../data/UnityEyesTrain',
                       'folder containing UnityEyes')
tf.flags.DEFINE_string('path_validation_unity', '../data/UnityEyesVal',
                       'folder containing UnityEyes')
tf.flags.DEFINE_string('path_validation_mpii', '../data/MPIIFaceGaze/single-eye_zhang.h5',
                       'file containing MPIIGaze')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('n_steps', 100000,
                        'number of steps to train. Half of the steps will be trained with a fix learning rate, the second half with linearly decaying LR.')
tf.flags.DEFINE_string('data_format', 'NHWC',
                       'NHWC or NCHW. default: NHWC')  # Important: This implementation does not yet support NCHW, so stick to NHWC!


def get_loss(path, image_size, gazenet, mode, regulariser=None):
    datasets = {
        Mode.TRAIN_UNITY: UnityDataset,
        Mode.VALIDATION_UNITY: UnityDataset,
        Mode.VALIDATION_MPII: MPIIDataset
    }
    # Prepare validation
    dataset = datasets[mode]
    iterator = dataset(path, image_size, FLAGS.batch_size, shuffle=True).get_iterator()
    is_training = mode == Mode.TRAIN_UNITY
    gaze_dict_validation, loss_validation = gazenet.get_loss(
        iterator, is_training=is_training, mode=mode, regulariser=regulariser)
    return loss_validation


def perform_validation_step(sess, losses_validation, summary_op, step, train_writer):
    loss_unity, loss_mpii, summary = (
        sess.run(
            [*losses_validation,
             summary_op],
        )
    )
    logging.info('-----------TEST at Step %d:-------------' % step)
    logging.info('  Time: {}'.format(
        datetime.now().strftime('%b-%d-%I%M%p-%G')))
    logging.info('  loss Unity     : {}'.format(loss_unity))
    logging.info('  loss MPIIGaze  : {}'.format(loss_mpii))

    train_writer.add_summary(summary, step)
    train_writer.flush()


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
            regulariser = tf.contrib.layers.l2_regularizer(scale=FLAGS.regularisation_lambda)
            loss_train = get_loss(FLAGS.path_train, image_size, gazenet, mode=Mode.TRAIN_UNITY, regulariser=regulariser)
            optimizers = gazenet.optimize(loss_train)

            loss_validation_unity = get_loss(FLAGS.path_validation_unity, image_size, gazenet, mode=Mode.VALIDATION_UNITY)
            loss_validation_mpii = get_loss(
                FLAGS.path_validation_mpii, image_size, gazenet, mode=Mode.VALIDATION_MPII)

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
                    # coord.
                    logging.info('Step {} -->  Time: {}    loss:  {}'.format(
                        step,
                        datetime.now().strftime('%b-%d-%I%M%p-%G'),
                        loss_value)
                    )

                if step >= 0 and step % 1000 == 0:
                    perform_validation_step(sess, [loss_validation_unity,
                        loss_validation_mpii], summary_op, step, train_writer)

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
