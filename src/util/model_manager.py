import logging
import os, argparse, shutil

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


class ModelManager:
    INPUT_NAME = "input"
    OUTPUT_NAME = "output"

    def __init__(self, save_folder, input_dimensions):
        self.save_folder = save_folder
        self.input_dimensions = input_dimensions

    def _remove_folder_if_exists(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)

    def save_model(self, sess, model):
        input_placeholder = tf.placeholder(tf.float32,
                                     shape=self.input_dimensions,
                                     name=self.INPUT_NAME)
        output_placeholder = model.sample(input_placeholder)
        output_placeholder = tf.identity(output_placeholder,
                                         name=self.OUTPUT_NAME)

        # simple_save creates a new folder and fails if the folder already
        # exists
        self._remove_folder_if_exists(self.save_folder)

        tf.saved_model.simple_save(
            sess,
            self.save_folder,
            {'in': input_placeholder},
            {'out': output_placeholder}
        )

        logging.info("Model saved in folder: {}".format(self.save_folder))

    def load_model(self, sess, graph):
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            self.save_folder,
        )
        in_tensor = graph.get_tensor_by_name("{}:0".format(self.INPUT_NAME))
        out_tensor = graph.get_tensor_by_name("{}:0".format(self.OUTPUT_NAME))

        logging.info("Model loaded from folder: {}".format(self.save_folder))
        return in_tensor, out_tensor
"""
graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        input_dimensions = [2, 72, 120, 3]

        input_image = np.random.randint(0, 10, input_dimensions)

        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
        '../checkpoints_gazenet/20181226-1725_debug_gazenet_m2u/saved_model',
        )
        in_tensor = graph.get_tensor_by_name('input_image:0')
        out_tensor = graph.get_tensor_by_name('output_gaze:0')

        r = sess.run(out_tensor, feed_dict={
            in_tensor: input_image
        })
        print(r)
"""




# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph

dir = os.path.dirname(os.path.realpath(__file__))


def freeze_graph(model_dir, output_node_names):
    """Extract the sub graph defined by the output nodes and convert
    all its variables into constant
    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names,
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        # We import the meta graph in the current default Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta',
                                           clear_devices=clear_devices)

        # We restore the weights
        saver.restore(sess, input_checkpoint)

        print([n.name for n in tf.get_default_graph().as_graph_def().node])

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),
            # The graph_def is used to retrieve the nodes
            output_node_names.split(",")
            # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_def


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="",
                        help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="",
                        help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    freeze_graph(args.model_dir, args.output_node_names)