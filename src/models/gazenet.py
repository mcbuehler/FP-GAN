import tensorflow as tf

import util.ops as ops
from models.base_gazenet import BaseGazeNet
from util.enum_classes import Mode
from util.model_utils import restore_model

"""
Original architecture:
Input (35x55)
-> Conv3x3 (f=32)
-> Conv3x3 (f=32)
-> Conv3x3 (f=64)
-> MaxPool3x3 (stride=2)
-> Conv3x3 (f=80)
-> Conv3x3 (f=192)
-> Maxpool2x2 (stride=2)
-> FC9600
-> FC1000
-> FC3
-> l2 normalisation  // ommitted in our case
-> l2 loss

Hyper-parameters:
LR = =.001
batch_size=512
"""


class GazeNet(BaseGazeNet):
    is_training_tensor = None
    variables = None

    def forward(self, input, mode, is_training=True):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1
            (default 1x5x5x1) filled with 0.9 if real, 0.0 if fake
        """
        self.is_training_tensor = tf.placeholder_with_default(
            is_training, shape=[], name='is_training')

        with tf.variable_scope(self.name, reuse=self.reuse):
            # convolution layers
            c32_1 = ops.conv3x3(input, k=32, stride=2, reuse=self.reuse,
                                norm=None,
                                is_training=self.is_training_tensor,
                                name="c32_1",
                                mode=mode)
            c32_2 = ops.conv3x3(c32_1, k=32, reuse=self.reuse, norm=self.norm,
                                is_training=self.is_training_tensor,
                                name="c32_2",
                                mode=mode)
            c64 = ops.conv3x3(c32_2, k=64, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training_tensor, name="c64",
                              mode=mode)
            maxpool3x3 = ops.maxpool(c64, 3, name="maxpool3x3", stride=2,
                                     reuse=self.reuse)
            c80 = ops.conv3x3(maxpool3x3, k=80, reuse=self.reuse,
                              norm=self.norm,
                              is_training=self.is_training_tensor, name="c80",
                              mode=mode)
            c192 = ops.conv3x3(c80, k=192, reuse=self.reuse, norm=self.norm,
                               is_training=self.is_training_tensor,
                               name="c192",
                               mode=mode)
            maxpool2x2 = ops.maxpool(c192, 2, name="maxpool2x2", stride=2,
                                     reuse=self.reuse)
            flattened = tf.contrib.layers.flatten(maxpool2x2)
            fc9600 = ops.dense(flattened, d=9600, name="fc9600",
                               reuse=self.reuse, mode=mode)
            fc1000 = ops.dense(fc9600, d=1000, name="fc1000", reuse=self.reuse,
                               mode=mode)
            out = ops.last_dense(
                fc1000, name="out", reuse=self.reuse
            )

        # This is used to easily access the trainable variables for this model
        self.variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return out


class GazeNetInference:
    """
    Run inference on a GazeNet
    """
    def __init__(self, sess, checkpoint_path, batch_size, image_size, norm, normalise_gaze, name):
        self.sess = sess
        self.checkpoint_path = checkpoint_path
        self.gazenet = GazeNet(
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=None,
            beta1=None,
            beta2=None,
            norm=norm,
            normalise_gaze=normalise_gaze,
            name=name
        )
        self.in_tensor, self.out_tensor = self.build_model()

    def build_model(self):
        """
        Create sample tensors for a GazeNet
        Returns:

        """
        in_tensor = tf.placeholder(tf.float32,
                                   (self.gazenet.batch_size,
                                    *self.gazenet.image_size, 1)
                                   )
        out_tensor = self.gazenet.forward(in_tensor, mode=Mode.TEST,
                                     is_training=False)
        return in_tensor, out_tensor

    def predict_gaze(self, images_preprocessed):
        """
        Run the predictions. This loads the model from the checkpoint and then
        runs a feed-forward for the preprocessed images.
        Args:
            images_preprocessed: np.array of preprocessed images

        Returns: gaze predictions for input images

        """
        restore_model(self.checkpoint_path, self.sess, self.gazenet.name)
        gaze_pred = self.sess.run(self.out_tensor,
                             feed_dict={self.in_tensor: images_preprocessed})
        return gaze_pred
