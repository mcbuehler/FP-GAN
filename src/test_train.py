import logging
import os

import numpy as np
import tensorflow as tf
from datetime import datetime

from models.model import CycleGAN
from input.reader import UnityReader, MPIIGazeReader
from util.utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('image_width', 120, 'image size, default: 256')
tf.flags.DEFINE_integer('image_height', 72, 'image size, default: 256')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
                        'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_integer('lambda2', 10,
                        'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
                        'number of gen filters in first conv layer, default: 64')

tf.flags.DEFINE_string('X', '../data/UnityEyes',
                       'X tfrecords file for training, default: ../data/tfrecords/apple.tfrecords')
tf.flags.DEFINE_string('Y', '../data/MPIIFaceGaze/single-eye_zhang.h5',
                       'Y tfrecords file for training, default: ../data/tfrecords/orange.tfrecords')
tf.flags.DEFINE_string('load_model', None,
                       'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')


def train():
    if FLAGS.load_model is not None:
        checkpoints_dir = "../checkpoints/" + FLAGS.load_model.lstrip(
            "checkpoints/")
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "../checkpoints/{}".format(current_time)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass

    graph = tf.Graph()

    image_size = [FLAGS.image_height, FLAGS.image_width]

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        step = 0

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            X_reader = UnityReader(FLAGS.X, name='X',
                                   image_size=image_size,
                                   batch_size=10,
                                   tf_session=sess)
            Y_reader = MPIIGazeReader(FLAGS.Y, name='X',
                                   image_size=image_size,
                                   batch_size=1,
                                   tf_session=sess)
            while True:
                # get previously generated images
                # # fake_y_val, fake_x_val = sess.run([fake_y, fake_x])
                #
                # x = X_reader.feed()
                # from util import utils
                # # x["eye"] = utils.batch_convert2int(x["eye"])
                # x_run = sess.run(x)
                # print(x_run["path"])
                # from matplotlib import pyplot as plt
                # image = x_run["eye"][0]
                # print(image.shape)
                # plt.imshow(image)
                # plt.show()

                y = Y_reader.feed()
                from util import utils
                y = utils.batch_convert2int(y)
                # x["eye"] = utils.batch_convert2int(x["eye"])
                y_run = sess.run(y)
                from matplotlib import pyplot as plt
                print(y_run.shape)
                plt.imshow(y_run[0])
                plt.show()

                #hist, bin_edges = np.histogram(image, bins=20)
                #plt.plot(bin_edges[1:], hist)
                #plt.show()
                #break

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # save_path = saver.save(sess, checkpoints_dir + "/model.ckpt",
            #                        global_step=step)
            # logging.info("Model saved in file: %s" % save_path)
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
