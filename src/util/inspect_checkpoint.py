import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = tf.train.latest_checkpoint('../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw/')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='gazenet_u_augmented_bw/out/weights', all_tensor_names=False)

from models.gazenet import GazeNet
from run.train_fpgan import restore_model

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        gazenet = GazeNet(
            batch_size=1,
            image_size=[36, 60],
            learning_rate=None,
            beta1=None,
            beta2=None,
            name="gazenet_u_augmented_bw",
            normalise_gaze=False,  # normalise_gaze,
            norm="batch",
        )


        t_in = tf.placeholder(shape=[1, 36, 60, 1], name='placeholder', dtype=tf.float32)
        t_out = gazenet.forward(t_in, is_training=False, mode="TEST")

        t = graph.get_tensor_by_name('gazenet_u_augmented_bw/out/weights:0')

        sess.run(tf.global_variables_initializer())

        a = sess.run(t)
        restore_model('../checkpoints_gazenet/20190107-1032_gazenet_u_augmented_bw/', sess)

        t_restored = graph.get_tensor_by_name('gazenet_u_augmented_bw/out/weights:0')
        b = sess.run(t_restored)
        print()
