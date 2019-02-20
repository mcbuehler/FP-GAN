import logging
import os
import shutil
from util.evaluation import get_validations
from input.dataset_manager import DatasetManager
from util.config_loader import Config
import tensorflow as tf
from datetime import datetime

from models.gazenet import GazeNet
from util.enum_classes import Mode

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')


def train():
    # Load the config variables
    cfg = Config(FLAGS.config, FLAGS.section)
    # Variables used for both directions
    batch_size = cfg.get('batch_size')
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
    path_validation_within = cfg.get('path_validation_within')
    path_validation_unity = cfg.get('path_validation_unity')
    path_validation_mpii = cfg.get('path_validation_mpii')
    dataset_class_train = cfg.get('dataset_class_train')
    dataset_class_validation_unity = cfg.get('dataset_class_validation_unity')
    dataset_class_validation_mpii = cfg.get('dataset_class_validation_mpii')
    do_augmentation = cfg.get('augmentation')
    rgb = cfg.get('rgb')
    normalise_gaze = cfg.get('normalise_gaze')
    filter_gaze = cfg.get('filter_gaze')
    # Indicates whether we are loading an existing model
    # or train a new one. Will be set to true below if we load an existing model.
    load_model = checkpoints_dir is not None and checkpoints_dir != ""

    checkpoint_dir_name = "checkpoints_gazenet"
    if not load_model:
        # We are not loading a saved model
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "../"+checkpoint_dir_name+"/{}_{}".format(current_time, model_name)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass

    logging.info("Checkpoint directory: {}".format(checkpoints_dir))

    graph = tf.Graph()

    # Solve memory issues
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=graph, config=config) as sess:
        with graph.as_default():
            gazenet = GazeNet(
                batch_size=batch_size,
                image_size=image_size,
                norm=norm,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                name=model_name,
                normalise_gaze=normalise_gaze
            )

            # Prepare training
            if cfg.get('use_regulariser'):
                logging.info("Using a l2 regulariser")
                regulariser = tf.contrib.layers.l2_regularizer(scale=cfg.get('regularisation_lambda'))
            else:
                regulariser = None

            train_iterator = DatasetManager.get_dataset_iterator_for_path(
                path_train, image_size, batch_size,
                shuffle=True, repeat=True, do_augmentation=do_augmentation,
                dataset_class=dataset_class_train, rgb=rgb,
                normalise_gaze=normalise_gaze, filter_gaze=filter_gaze
            )
            _, loss_train = gazenet.get_loss(
                train_iterator, is_training=True,
                mode=Mode.TRAIN, regulariser=regulariser)
            optimizers = gazenet.optimize(loss_train)

            # Summaries and saver
            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
            # We need to save global variables because we want to save
            # variables from batch_norm layers
            saver = tf.train.Saver(tf.global_variables())

        # Validation graphs
        # We do this after tf.summary_merge_all because we don't want to create
        # summaries for every step
        all_validations = get_validations(
            gazenet, path_validation_within, dataset_class_train, path_validation_unity, dataset_class_validation_unity, path_validation_mpii, dataset_class_validation_mpii, image_size, batch_size, rgb=rgb, normalise_gaze=normalise_gaze, filter_gaze=filter_gaze
        )

        if load_model:# and False:
            logging.info("Restoring from checkpoint directory: {}".format(checkpoints_dir))
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split("-")[-1].split(".")[0])
        else:
            logging.info("Training new model. Checkpoint directory: {}".format(checkpoints_dir))
            # We copy config file
            shutil.copyfile(FLAGS.config, os.path.join(checkpoints_dir, "{}__{}.ini".format(model_name, FLAGS.section)))
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

                    # if step > 0 and step % 5000 == 0:
                if 0 <= step < n_steps and step % 5000 == 0:
                    # model_manager.save_model(sess, gazenet)
                    save_path = saver.save(sess,
                                           checkpoints_dir + "/model.ckpt",
                                           global_step=step)
                    for validation in all_validations:
                        validation.run(sess,
                                                           step,
                                                           train_writer,
                                                           n_batches=15)
                step += 1
        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # model_manager.save_model(sess, gazenet)
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt",
                                   global_step=step)
            logging.info("Model saved in file: %s" % save_path)
            for validation in all_validations:
                validation.run(
                    sess,
                    step,
                    train_writer
                )
            # When done, ask the threads to stop.
            coord.request_stop()
            # coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
