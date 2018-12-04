import tensorflow as tf


def conv(name, l, channel, stride):
    return tf.layers.conv2d(l, filters=channel, kernel_size=3,
                            strides=stride,
                            padding='same', name=name)


def add_layer(name, l, growth_rate, is_training=True):
    """
    Adds BN, ReLU and Conv layer
    :param name: will be used as variable scope
    :param l: input tensor
    :return:
    """
    with tf.variable_scope(name):
        c = tf.layers.batch_normalization(l, name='bn1',
                                          training=is_training)
        c = tf.nn.relu(c)
        c = conv('conv1', c, growth_rate, 1)
        l = tf.concat([c, l], 3)
    return l


def add_transition(name, l, is_training=True):
    """
    Adds a transition layer. Consists of BN, ReLU, Conv, ReLU, AvgPooling
    :param name: variable scope
    :param l: input tensor
    :return:
    """
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name):
        l = tf.layers.batch_normalization(l, name='bn1',
                                          training=is_training)
        l = tf.nn.relu(l)
        l = tf.layers.conv2d(l, filters=in_channel, strides=1,
                             kernel_size=1, padding='same', name='conv1')
        l = tf.nn.relu(l)
        layer = tf.layers.AveragePooling2D(name='pool', padding='same',
                                           strides=2,
                                           pool_size=2)
        l = layer.apply(l, scope=tf.get_variable_scope())
    return l


def global_average_pooling(x, data_format='channels_last', name=None):
    """
    Global average pooling as in the paper `Network In Network
    <http://arxiv.org/abs/1312.4400>`_.
    Args:
        x (tf.Tensor): a 4D tensor.
    Returns:
        tf.Tensor: a NC tensor named ``output``.
    """
    assert x.shape.ndims == 4
    axis = [1, 2] if data_format == 'channels_last' else [2, 3]
    return tf.reduce_mean(x, axis, name=name)
