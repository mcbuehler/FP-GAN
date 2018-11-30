import logging
import os

import tensorflow as tf
from datetime import datetime

from models.densegazenet import DenseGazeNet
from datasources.unityeyes import UnityEyes

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 32, 'batch size, default: 512')
tf.flags.DEFINE_integer('image_width', 120, 'default: 120')
tf.flags.DEFINE_integer('image_height', 72, 'default: 72')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')

tf.flags.DEFINE_float('learning_rate', 0.001,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.9,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('beta2', 0.999,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_string('path_train', '../data/UnityEyesTrain',
                       'folder containing UnityEyes')
tf.flags.DEFINE_string('path_validation', '../data/UnityEyesVal',
                       'folder containing UnityEyes')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
tf.flags.DEFINE_integer('n_steps', 100000,
                        'number of steps to train. Half of the steps will be trained with a fix learning rate, the second half with linearly decaying LR.')
tf.flags.DEFINE_string('data_format', 'NHWC',
                       'NHWC or NCHW. default: NHWC')  # Important: This implementation does not yet support NCHW, so stick to NHWC!


def get_image_queue(DataSource, path, is_training, sess, batch_size, image_size):
    if is_training:
        image_queue = DataSource(sess, batch_size, path,
                            eye_image_shape=image_size,
                            data_format="NHWC", shuffle=True)
    else:
        image_queue = DataSource(sess, batch_size, path,
                            eye_image_shape=image_size,
                            data_format="NHWC", shuffle=False,
                                 testing=True)

    image_queue.create_and_start_threads()
    return image_queue


def perform_validation_step(sess, loss_validation, summary_op, step, train_writer):
    loss_value, summary = (
        sess.run(
            [loss_validation,
             summary_op],
        )
    )
    logging.info('-----------TEST at Step %d:-------------' % step)
    logging.info('  Time: {}'.format(
        datetime.now().strftime('%b-%d-%I%M%p-%G')))
    logging.info('  loss   : {}'.format(loss_value))

    train_writer.add_summary(summary, step)
    train_writer.flush()

def train():
    checkpoint_dir_name = "checkpoints_densegazenet"
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
            gazenet = DenseGazeNet(
                batch_size=FLAGS.batch_size,
                image_size=image_size,
                norm=FLAGS.norm,
                learning_rate=FLAGS.learning_rate,
                beta1=FLAGS.beta1,
                beta2=FLAGS.beta2,
                tf_session=sess,
                name="densegazenet"
            )

            # Prepare training
            image_queue_train = get_image_queue(UnityEyes, FLAGS.path_train, is_training=True, sess=sess, batch_size=FLAGS.batch_size, image_size=image_size)
            gaze_dict_train, loss_train = gazenet.get_loss(image_queue_train.output_tensors)
            optimizers = gazenet.optimize(loss_train, n_steps=FLAGS.n_steps)

            # Prepare validation
            image_queue_validation = get_image_queue(UnityEyes, FLAGS.path_train, is_training=False, sess=sess, batch_size=FLAGS.batch_size, image_size=image_size)
            gaze_dict_validation, loss_validation = gazenet.get_loss(
                image_queue_validation.output_tensors, is_training=False)

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
            sess.run(tf.global_variables_initializer())
            step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

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

                n_info_steps = 100
                if step % n_info_steps == 0:
                    # coord.
                    logging.info('Step {} -->  Time: {}    loss:  {}'.format(
                        step,
                        datetime.now().strftime('%b-%d-%I%M%p-%G'),
                        loss_value)
                    )

                if step % 1000 == 0:
                    perform_validation_step(sess, loss_validation, summary_op, step, train_writer)

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
            coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
