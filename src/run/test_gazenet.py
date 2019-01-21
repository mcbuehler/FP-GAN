import logging
from util.evaluation import Test
from util.config_loader import Config
import tensorflow as tf

from models.gazenet import GazeNet
from util.enum_classes import Mode

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('config', None, 'input configuration')
tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')


def test():
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
    path_test = cfg.get('path_test')
    dataset_class_test = cfg.get('dataset_class_test')
    rgb = cfg.get('rgb')
    normalise_gaze = cfg.get('normalise_gaze')
    filter_gaze = cfg.get('filter_gaze')

    logging.info("Checkpoint directory: {}".format(checkpoints_dir))

    # Solve memory issues
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
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
        test_step = Test(
            gazenet,
            Mode.TEST,
            path_test,
            image_size,
            batch_size,
            dataset_class_test,
            rgb=rgb,
            normalise_gaze=normalise_gaze,
            filter_gaze=filter_gaze
        )

        saver = tf.train.Saver()
        logging.info("Restoring from checkpoint directory: {}".format(checkpoints_dir))
        # checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
        # meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
        # restore = tf.train.import_meta_graph(meta_graph_path)
        saver.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        # step = int(meta_graph_path.split("-")[2].split(".")[0])
        step = 0

        test_step.run(sess, step, write_folder=checkpoints_dir)


def main(unused_argv):
    test()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
