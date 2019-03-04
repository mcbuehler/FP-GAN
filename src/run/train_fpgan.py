"""
Training script for FP-GAN
Make sure to include a valid configuration.

Example running command:
CUDA_VISIBLE_DEVICES=6 python run/train_fpgan.py --config ../config/fpgan_ege.ini --section EGE_RHP
"""

import logging
import os

import tensorflow as tf
from datetime import datetime

from models.model import CycleGAN
from util.utils import ImagePool
from util.config_loader import Config
from util.model_utils import restore_model


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration section')


def train():
    # Load the config variables
    cfg = Config(FLAGS.config, FLAGS.section)

    # Variables used for both directions
    gan_name = cfg.get('gan_name')
    batch_size = cfg.get('batch_size')
    image_size = [cfg.get('image_height'),
                  cfg.get('image_width')]
    use_lsgan = cfg.get('use_lsgan')
    norm_gan = cfg.get('norm')
    rgb = cfg.get('rgb')
    lambda1 = cfg.get('lambda1')
    lambda2 = cfg.get('lambda2')
    lambda_identity = cfg.get('lambda_identity')
    lambda_gaze = cfg.get('lambda_gaze')
    lambda_landmarks = cfg.get('lambda_landmarks')
    learning_rate = cfg.get('learning_rate')

    beta1 = cfg.get('beta1')
    pool_size = cfg.get('pool_size')
    ngf = cfg.get('ngf')
    S = cfg.get('S')
    R = cfg.get('R')
    checkpoints_dir = cfg.get('checkpoint_folder')
    n_steps = cfg.get('n_steps')
    filter_gaze = cfg.get('filter_gaze')

    # If we do not want to train a new model, but rather continue
    # training one, we have a non-empty checkpoints_dir
    load_model = checkpoints_dir is not None and checkpoints_dir != ""

    lambdas_features = {
        'identity': lambda_identity,
        'gaze': lambda_gaze,
        'landmarks': lambda_landmarks
    }

    ege_config = {
        'norm': cfg.get('ege_norm'),
        'normalise_gaze': cfg.get('normalise_gaze'),
        'name': cfg.get('ege_name')
    }

    # Print an info every n_info_steps
    n_info_steps = 100

    if not load_model:
        # This will be included in the model checkpoint directory for
        # better traceability of experiments
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = "../checkpoints/{}_{}".format(current_time, gan_name)
    try:
        os.makedirs(checkpoints_dir)
    except os.error:
        pass

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        with graph.as_default():
            # Build the model
            cycle_gan = CycleGAN(
                S_train_file=S,
                R_train_file=R,
                batch_size=batch_size,
                image_size=image_size,
                use_lsgan=use_lsgan,
                norm=norm_gan,
                rgb=rgb,
                lambda1=lambda1,
                lambda2=lambda2,
                lambdas_features=lambdas_features,
                learning_rate=learning_rate,
                beta1=beta1,
                ngf=ngf,
                tf_session=sess,
                filter_gaze=filter_gaze,
                ege_config=ege_config
            )

        # Get all losses and build optimizer
        G_loss, D_R_loss, F_loss, D_S_loss, fake_r, fake_s = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_R_loss, F_loss, D_S_loss,
                                        n_steps=n_steps)

        if load_model:
            step = restore_model(checkpoints_dir, sess)
        else:
            sess.run(tf.global_variables_initializer())
            step = 0

        saver = tf.train.Saver()

        # If we apply customised feature losses, we need to build and restore
        # the feature estimation networks, too.
        if lambdas_features['gaze'] > 0:
            # load eye gaze feature model
            ege_model_path = cfg.get('ege_path')
            ege_model_name = cfg.get('ege_name')
            restore_model(ege_model_path, sess=sess, variables_scope=ege_model_name)
        if lambdas_features['landmarks'] > 0:
            lm_model_path = cfg.get('lm_path')
            # lm_model_name = cfg.get('lm_name')
            # load landmarks feature model
            restore_model(lm_model_path, sess=sess, is_model_path=True, variables_scope='hourglass')

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)

        coord = tf.train.Coordinator()
        # Collect errors
        errors = list()

        # We update the discriminator using a history of images.
        # This has been shown to stabilise training.
        fake_R_pool = ImagePool(pool_size)
        fake_S_pool = ImagePool(pool_size)

        while step < n_steps and not coord.should_stop():
            try:
                fake_r_val, fake_s_val = sess.run([fake_r, fake_s])

                # Do step with summaries
                _, G_loss_val, D_R_loss_val, F_loss_val, D_S_loss_val, summary = (
                    sess.run(
                        [optimizers, G_loss, D_R_loss, F_loss, D_S_loss,
                         summary_op],
                        feed_dict={
                            cycle_gan.fake_r: fake_R_pool.query(
                                fake_r_val),
                            cycle_gan.fake_s: fake_S_pool.query(
                                fake_s_val)}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if step % n_info_steps == 0:
                    logging.info('-----------Step %d:-------------' % step)
                    logging.info('  Time: {}'.format(
                        datetime.now().strftime('%b-%d-%I%M%p-%G')))
                    logging.info('  G_loss   : {}'.format(G_loss_val))
                    logging.info('  D_Y_loss : {}'.format(D_R_loss_val))
                    logging.info('  F_loss   : {}'.format(F_loss_val))
                    logging.info('  D_X_loss : {}'.format(D_S_loss_val))

                if step > 0 and step % 10000 == 0:
                    save_path = saver.save(sess,
                                           checkpoints_dir + "/model.ckpt",
                                           global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1
            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except ValueError as e:
                errors.append(e)
                logging.warning("Value error. Skipping batch. Total errors: {}".format(len(errors)))
            except Exception as e:
                logging.warning("Unforeseen error. Requesting stop...")
                coord.request_stop(e)

        save_path = saver.save(sess, checkpoints_dir + "/model.ckpt",
                               global_step=step)
        logging.info("Model saved in file: %s" % save_path)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
