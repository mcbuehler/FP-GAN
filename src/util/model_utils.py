import tensorflow as tf
import re
import logging




def restore_model(path, sess, variables_scope=None, is_model_path=False):
    variables_can_be_restored = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=variables_scope))
    logging.info("Loading model from '{}'".format(path))
    if is_model_path:
        # Path is something like ...model-76739
        restore_path = path
    else:
        # Path is checkpoint folder, e.g. 20190107-1032_gazenet_u_augmented_bw/
        checkpoint = tf.train.get_checkpoint_state(path)
        restore_path = checkpoint.model_checkpoint_path
    # print("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node]))
    restore = tf.train.Saver(variables_can_be_restored)
    restore.restore(sess, restore_path)
    step = int(re.sub(r'[^\d]', '', restore_path.split('-')[-1]))
    return step