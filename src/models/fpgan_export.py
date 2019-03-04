import tensorflow as tf

from models.model import CycleGAN
from util.config_loader import Config


class GeneratorExport:
    """ Freeze variables and convert 2 generator networks to 2 GraphDef files.
    This makes file size smaller and can be used for inference in production.
    """
    def __init__(self, checkpoint_dir, image_size, batch_size, norm, ngf, rgb):
        """
        Args:
            checkpoint_dir: Where to load the model from
            image_size: (height, width)
            batch_size: will need to be the same for inference
            norm: instance or batch
            ngf: number of filters in first layer
            rgb: RGB images or gray-scale
        """
        self.checkpoint_dir = checkpoint_dir
        self.norm = norm
        self.ngf = ngf
        self.image_size = image_size
        self.batch_size = batch_size
        self.rgb = rgb

    def run(self, model_name, S2R=True):
        """
        Run export for given model. The direction is given by S2R
        Args:
            model_name: name to use when writing graph
            S2R: Unity2MPII (corresponds to synthetic-to-real)

        Returns:
        """
        graph = tf.Graph()
        if self.rgb:
            input_dimensions = [self.batch_size, *self.image_size, 3]
        else:
            input_dimensions = [self.batch_size, *self.image_size, 1]

        with graph.as_default():
            cycle_gan = CycleGAN(ngf=self.ngf, norm=self.norm,
                                 image_size=self.image_size, rgb=self.rgb)

            input_image = tf.placeholder(tf.float32, shape=input_dimensions,
                                         name='input_image')
            cycle_gan.model(fake_input=True)
            if S2R:
                output_image = cycle_gan.G.sample(input_image)
            else:
                output_image = cycle_gan.F.sample(input_image)

            output_image = tf.identity(output_image, name='output_image')
            saver = tf.train.Saver()

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)
            print("Latest checkpoint: {}".format(latest_ckpt))

            saver.restore(sess, latest_ckpt)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(), [output_image.op.name])

            tf.train.write_graph(output_graph_def, self.checkpoint_dir,
                                 model_name, as_text=False)


def main(unused_argv):
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
    print('Exporting S2R model...')
    default_args = dict(
        checkpoint_dir=cfg.get("checkpoint_folder"),
        image_size=[cfg.get('image_height'),
                    cfg.get('image_width')],
        batch_size=cfg.get("batch_size_inference"),
        norm=cfg.get("norm"),
        ngf=cfg.get("ngf")
    )

    generator_export = GeneratorExport(**default_args)

    generator_export.run(
        model_name=cfg.get("model_name_s2r"),
        S2R=True)

    print('Exporting R2S model...')
    generator_export.run(
        model_name=cfg.get("model_name_r2s"),
        S2R=False)


if __name__ == '__main__':
    tf.app.run()
