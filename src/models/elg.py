"""ELG architecture."""
import numpy as np
import scipy
import tensorflow as tf


def _tf_mse(x, y):
    """Tensorflow call for mean-squared error."""
    return tf.reduce_mean(tf.squared_difference(x, y))


class ELG:
    """ELG architecture as introduced in [Park et al. ETRA'18]."""

    def __init__(self, tensorflow_session=None, first_layer_stride=2,
                 num_modules=3, num_feature_maps=64, **kwargs):
        """Specify ELG-specific parameters."""
        self._hg_first_layer_stride = first_layer_stride
        self._hg_num_modules = num_modules
        self._hg_num_feature_maps=num_feature_maps

        # Call parent class constructor
        # super().__init__(tensorflow_session, **kwargs)

    _hg_first_layer_stride = 2
    _hg_num_modules = 4
    _hg_num_feature_maps = 64
    _hg_num_landmarks = 18
    _hg_num_residual_blocks = 1

    _data_format_longer = "channels_last"  # "channels_first" #
    _data_format = "NHWC"  # NCHW"
    use_batch_statistics = False

    reuse = tf.AUTO_REUSE

    # @property
    # def identifier(self):
    #     """Identifier for model based on data sources and parameters."""
    #     first_data_source = next(iter(self._train_data.values()))
    #     input_tensors = first_data_source.output_tensors
    #     if self._data_format == 'NHWC':
    #         _, eh, ew, _ = input_tensors['eye'].shape.as_list()
    #     else:
    #         _, _, eh, ew = input_tensors['eye'].shape.as_list()
    #     return 'ELG_i%dx%d_f%dx%d_n%d_m%d_rgb' % (
    #         ew, eh,
    #         int(ew / self._hg_first_layer_stride),
    #         int(eh / self._hg_first_layer_stride),
    #         self._hg_num_feature_maps, self._hg_num_modules,
    #     )

    # def train_loop_pre(self, current_step):
    #     """Run this at beginning of training loop."""
    #     Set difficulty of training data
        # data_source = next(iter(self._train_data.values()))
        # data_source.set_difficulty(min((1. / 1e6) * current_step, 1.))

    def build_model(self, input_tensors):
        """Build model."""
        # data_source = next(iter(data_sources.values()))
        # input_tensors = data_source.output_tensors
        # x = input_tensors['eye']
        x = input_tensors
        x_original = x

        # y1 = input_tensors['heatmaps'] if 'heatmaps' in input_tensors else None
        # y2 = input_tensors['landmarks'] if 'landmarks' in input_tensors else None
        # y3 = input_tensors['radius'] if 'radius' in input_tensors else None

        outputs = {}
        loss_terms = {}
        metrics = {}

        with tf.variable_scope('hourglass', reuse=self.reuse):
            # TODO: Find better way to specify no. landmarks
            # if y1 is not None:
            #     if self._data_format == 'NCHW':
            #         self._hg_num_landmarks = y1.shape.as_list()[1]
            #     if self._data_format == 'NHWC':
            #         self._hg_num_landmarks = y1.shape.as_list()[3]
            # else:
            #     self._hg_num_landmarks = 18
            assert self._hg_num_landmarks == 18

            # Prepare for Hourglass by downscaling via conv
            with tf.variable_scope('pre'):
                n = self._hg_num_feature_maps
                x = self._apply_conv(x, num_features=n, kernel_size=7,
                                     stride=self._hg_first_layer_stride)
                x = tf.nn.relu(self._apply_bn(x))
                x = self._build_residual_block(x, n, 2*n, name='res1')
                x = self._build_residual_block(x, 2*n, n, name='res2')

            # Hourglass blocks
            x_prev = x
            for i in range(self._hg_num_modules):
                with tf.variable_scope('hg_%d' % (i + 1)):
                    x = self._build_hourglass(x, steps_to_go=4, num_features=self._hg_num_feature_maps)
                    x, h = self._build_hourglass_after(
                        x_prev, x, do_merge=(i < (self._hg_num_modules - 1)),
                    )
                    # if y1 is not None:
                    #     metrics['heatmap%d_mse' % (i + 1)] = _tf_mse(h, y1)
                    x_prev = x
            # if y1 is not None:
            #     loss_terms['heatmaps_mse'] = tf.reduce_mean([
            #         metrics['heatmap%d_mse' % (i + 1)] for i in range(self._hg_num_modules)
            #     ])
            x = h
            outputs['heatmaps'] = x

        # Soft-argmax
        x = self._calculate_landmarks(x)
        with tf.variable_scope('upscale', reuse=self.reuse):
            # Upscale since heatmaps are half-scale of original image
            x *= self._hg_first_layer_stride
            # if y2 is not None:
            #     metrics['landmarks_mse'] = _tf_mse(x, y2)
            outputs['landmarks'] = x

        outputs['eye_input'] = x_original

        return outputs, loss_terms, metrics

    def _apply_conv(self, tensor, num_features, kernel_size=3, stride=1):
        return tf.layers.conv2d(
            tensor,
            num_features,
            kernel_size=kernel_size,
            strides=stride,
            padding='SAME',
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            data_format=self._data_format_longer,
            name='conv',
        )

    def _apply_fc(self, tensor, num_outputs):
        return tf.layers.dense(
            tensor,
            num_outputs,
            use_bias=True,
            kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            bias_initializer=tf.zeros_initializer(),
            name='fc',
        )

    def _apply_pool(self, tensor, kernel_size=3, stride=2):
        tensor = tf.layers.max_pooling2d(
            tensor,
            pool_size=kernel_size,
            strides=stride,
            padding='SAME',
            data_format=self._data_format_longer,
            name='pool',
        )
        return tensor

    def _apply_bn(self, tensor):
        return tf.contrib.layers.batch_norm(
            tensor,
            scale=True,
            center=True,
            is_training=self.use_batch_statistics,
            trainable=True,
            data_format=self._data_format,
            updates_collections=None,
        )

    def _build_residual_block(self, x, num_in, num_out, name='res_block'):
        with tf.variable_scope(name):
            half_num_out = max(int(num_out/2), 1)
            c = x
            with tf.variable_scope('conv1'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=1, stride=1)
            with tf.variable_scope('conv2'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=half_num_out, kernel_size=3, stride=1)
            with tf.variable_scope('conv3'):
                c = tf.nn.relu(self._apply_bn(c))
                c = self._apply_conv(c, num_features=num_out, kernel_size=1, stride=1)
            with tf.variable_scope('skip'):
                if num_in == num_out:
                    s = tf.identity(x)
                else:
                    s = self._apply_conv(x, num_features=num_out, kernel_size=1, stride=1)
            x = c + s
        return x

    def _build_hourglass(self, x, steps_to_go, num_features, depth=1):
        with tf.variable_scope('depth%d' % depth):
            # Upper branch
            up1 = x
            for i in range(self._hg_num_residual_blocks):
                up1 = self._build_residual_block(up1, num_features, num_features,
                                                 name='up1_%d' % (i + 1))
            # Lower branch
            low1 = self._apply_pool(x, kernel_size=2, stride=2)
            for i in range(self._hg_num_residual_blocks):
                low1 = self._build_residual_block(low1, num_features, num_features,
                                                  name='low1_%d' % (i + 1))
            # Recursive
            low2 = None
            if steps_to_go > 1:
                low2 = self._build_hourglass(low1, steps_to_go - 1, num_features, depth=depth+1)
            else:
                low2 = low1
                for i in range(self._hg_num_residual_blocks):
                    low2 = self._build_residual_block(low2, num_features, num_features,
                                                      name='low2_%d' % (i + 1))
            # Additional residual blocks
            low3 = low2
            for i in range(self._hg_num_residual_blocks):
                low3 = self._build_residual_block(low3, num_features, num_features,
                                                  name='low3_%d' % (i + 1))
            # Upsample
            if self._data_format == 'NCHW':  # convert to NHWC
                low3 = tf.transpose(low3, (0, 2, 3, 1))
            up2 = tf.image.resize_bilinear(
                    low3,
                    up1.shape[1:3] if self._data_format == 'NHWC' else up1.shape[2:4],
                    align_corners=True,
                  )
            if self._data_format == 'NCHW':  # convert back from NHWC
                up2 = tf.transpose(up2, (0, 3, 1, 2))

        return up1 + up2

    def _build_hourglass_after(self, x_prev, x_now, do_merge=True):
        with tf.variable_scope('after'):
            for j in range(self._hg_num_residual_blocks):
                x_now = self._build_residual_block(x_now, self._hg_num_feature_maps,
                                                   self._hg_num_feature_maps,
                                                   name='after_hg_%d' % (j + 1))
            x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
            x_now = self._apply_bn(x_now)
            x_now = tf.nn.relu(x_now)

            with tf.variable_scope('hmap'):
                h = self._apply_conv(x_now, self._hg_num_landmarks, kernel_size=1, stride=1)

        x_next = x_now
        if do_merge:
            with tf.variable_scope('merge'):
                with tf.variable_scope('h'):
                    x_hmaps = self._apply_conv(h, self._hg_num_feature_maps, kernel_size=1, stride=1)
                with tf.variable_scope('x'):
                    x_now = self._apply_conv(x_now, self._hg_num_feature_maps, kernel_size=1, stride=1)
                x_next += x_prev + x_hmaps
        return x_next, h

    _softargmax_coords = None

    def _calculate_landmarks(self, x):
        """Estimate landmark location from heatmaps."""
        with tf.variable_scope('argsoftmax'):
            if self._data_format == 'NHWC':
                _, h, w, _ = x.shape.as_list()
            else:
                _, _, h, w = x.shape.as_list()
            if self._softargmax_coords is None:
                # Assume normalized coordinate [0, 1] for numeric stability
                ref_xs, ref_ys = np.meshgrid(np.linspace(0, 1.0, num=w, endpoint=True),
                                             np.linspace(0, 1.0, num=h, endpoint=True),
                                             indexing='xy')
                ref_xs = np.reshape(ref_xs, [-1, h*w])
                ref_ys = np.reshape(ref_ys, [-1, h*w])
                self._softargmax_coords = (
                    tf.constant(ref_xs, dtype=tf.float32),
                    tf.constant(ref_ys, dtype=tf.float32),
                )
            ref_xs, ref_ys = self._softargmax_coords

            # Assuming N x 18 x 45 x 75 (NCHW)
            beta = 1e2
            if self._data_format == 'NHWC':
                x = tf.transpose(x, (0, 3, 1, 2))
            x = tf.reshape(x, [-1, self._hg_num_landmarks, h*w])
            x = tf.nn.softmax(beta * x, axis=-1)
            lmrk_xs = tf.reduce_sum(ref_xs * x, axis=[2])
            lmrk_ys = tf.reduce_sum(ref_ys * x, axis=[2])

            # Return to actual coordinates ranges
            return tf.stack([
                lmrk_xs * (w - 1.0) + 0.5,
                lmrk_ys * (h - 1.0) + 0.5,
            ], axis=2)  # N x 18 x 2

