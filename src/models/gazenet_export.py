""" Freeze variables and convert 2 generator networks to 2 GraphDef files.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:

python run/export_fp_gan.py --config ../config/fpgan_basic.ini --section 20181207-1957
"""
from util.config_loader import Config
import tensorflow as tf
import logging

from models.gazenet import GazeNet


class GazeNetExport:
    def __init__(self, checkpoint_dir, image_size, batch_size, norm):
        self.checkpoint_dir = checkpoint_dir
        self.norm = norm
        self.image_size = image_size
        self.batch_size = batch_size

    def run(self, model_name):
        graph = tf.Graph()
        input_dimensions = [self.batch_size, *self.image_size, 3]

        with graph.as_default():
            model = GazeNet(
                batch_size=self.batch_size,
                image_size=self.image_size,
                norm=self.norm,
                name=model_name
            )

            input_image = tf.placeholder(tf.float32, shape=input_dimensions,
                                         name='input_image')
            output_gaze = model.sample(input_image)
            output_gaze = tf.identity(output_gaze, name='output_gaze')

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            restore = tf.train.import_meta_graph(meta_graph_path)
            latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
            logging.info("Latest checkpoint: {}".format(latest_ckpt))
            restore.restore(sess, latest_ckpt)
            step = int(meta_graph_path.split("-")[2].split(".")[0])

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [output_gaze.op.name])

            tf.train.write_graph(output_graph_def, self.checkpoint_dir,
                                 model_name, as_text=False)


def main():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_string('config', None, 'input configuration')
    tf.flags.DEFINE_string('section', 'DEFAULT', 'input configuration')

    if FLAGS.config is None:
        print("Please provide config file (--config PATH).")
        exit()

    # Load the config variables
    cfg_section = FLAGS.section
    cfg = Config(FLAGS.config, cfg_section)

    print("Exporting model from {}...".format(cfg.get("checkpoint_folder")))
    default_args = dict(
        checkpoint_dir=cfg.get("checkpoint_folder"),
        image_size=[cfg.get('image_height'),
                  cfg.get('image_width')],
        batch_size=cfg.get("batch_size_inference"),
        norm=cfg.get("norm")
    )

    generator_export = GazeNetExport(**default_args)
    generator_export.run(cfg.get('model_name_pb'))



if __name__ == '__main__':
    main()
