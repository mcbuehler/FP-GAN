import tensorflow as tf

import util.ops as ops
import util.gaze as gaze
import util.utils as utils
from base_gazenet import BaseGazeNet

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
-> l2 normalisation
-> l2 loss
LR = =.001
batch_size=512
"""


class GazeNet(BaseGazeNet):

    def forward(self, input, is_training=True):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        self.is_training = tf.placeholder_with_default(is_training, shape=[],
                                                       name='is_training')

        with tf.variable_scope(self.name, reuse=self.reuse):
            # convolution layers
            c32_1 = ops.conv3x3(input, k=32, stride=2, reuse=self.reuse, norm=None,
                                is_training=self.is_training, name="c32_1", summary=True)
            c32_2 = ops.conv3x3(c32_1, k=32, reuse=self.reuse, norm=self.norm,
                                is_training=self.is_training, name="c32_2")
            c64 = ops.conv3x3(c32_2, k=64, reuse=self.reuse, norm=self.norm,
                              is_training=self.is_training, name="c64")
            maxpool3x3 = ops.maxpool(c64, 3, name="maxpool3x3", stride=2,
                                     reuse=self.reuse)
            c80 = ops.conv3x3(maxpool3x3, k=80, reuse=self.reuse,
                              norm=self.norm,
                              is_training=self.is_training, name="c80")
            c192 = ops.conv3x3(c80, k=192, reuse=self.reuse, norm=self.norm,
                               is_training=self.is_training, name="c192")
            maxpool2x2 = ops.maxpool(c192, 2, name="maxpool2x2", stride=2,
                                     reuse=self.reuse)
            flattened = tf.contrib.layers.flatten(maxpool2x2)
            fc9600 = ops.dense(flattened, d=9600, name="fc9600",
                               reuse=self.reuse)
            fc1000 = ops.dense(fc9600, d=1000, name="fc1000", reuse=self.reuse)
            out = ops.last_dense(fc1000, name="out", reuse=self.reuse, use_sigmoid=self.use_sigmoid)
            out_normalised = tf.nn.l2_normalize(out, axis=1, name="l2_normalise")
        # What about a layer that adds a restriction on output?
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope=self.name)

        return out_normalised
