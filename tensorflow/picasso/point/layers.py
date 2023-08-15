import tensorflow as tf
import tensorflow_addons as tfa
import os, sys
# import .modules as modules
from .modules import *

def _filter_variable_(name, shape, stddev, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        use_xavier: bool, whether to use the Xavier(Glorot) normal initializer
    Returns:
        Variable Tensor
    """
    if use_xavier:
        initializer = tf.keras.initializers.GlorotNormal()
    else:
        initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)
    # variable is trainable in default.
    var = tf.Variable(initial_value=initializer(shape=shape),
                      trainable=True, name=name, dtype=tf.float32)
    return var


class BatchNorm(tf.Module):
    def __init__(self, bn_momentum=0.9, scope=''):
        super(BatchNorm,self).__init__()
        self.norm_fn = tf.keras.layers.BatchNormalization(axis=-1, momentum=bn_momentum,
                                    beta_regularizer=tf.keras.regularizers.l2(1.0),
                                    gamma_regularizer=tf.keras.regularizers.l2(1.0),
                                    name=scope+'/BN')
        # print(colored('batchNorm momentum: {}'.format(bn_momentum), 'red'))

    def __call__(self, data, is_training):
        return self.norm_fn(data, training=is_training)


class Dropout(tf.Module):

    def __init__(self, drop_rate=0.5):
        super(Dropout,self).__init__()
        self.DP = tf.keras.layers.Dropout(rate=drop_rate)

    def __call__(self, inputs, is_training):
        return self.DP(inputs, training=is_training)


class PCloudConv3d(tf.Module):
    """ 3D separable convolution with non-linear operation.
    Args:
        in_channels:  number of input  channels
        out_channels: number of output channels
        depth_multiplier: depth multiplier in the separable convolution
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
    Returns:
      Variable tensor
    """
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1,
                 scope='', use_xavier=True, stddev=1e-3,  with_bn=True,
                 activation_fn=tf.nn.relu, bn_momentum=0.9):
        super(PCloudConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scope = scope
        self.kernel_size = kernel_size
        self.multiplier = depth_multiplier
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        self.filter = {}
        if self.with_bn:
            self.BatchNorm = BatchNorm(bn_momentum=bn_momentum, scope=self.scope)
        self.reset_parameters()

    def reset_parameters(self):
        # depthwise kernel shape
        depthwise_kernel_shape = [self.kernel_size, self.in_channels, self.multiplier]
        self.filter['depthwise'] = _filter_variable_(self.scope+'/depthwise_weights',
                                       shape=depthwise_kernel_shape, stddev=self.stddev,
                                       use_xavier=self.use_xavier)
        # pointwise kernel shape
        kernel_shape = [self.in_channels*self.multiplier, self.out_channels]
        self.filter['pointwise'] = _filter_variable_(self.scope+'/pointwise_weights',
                                       shape=kernel_shape, use_xavier=self.use_xavier,
                                       stddev=self.stddev)
        # biases term
        self.filter['biases'] = tf.Variable(tf.zeros(self.out_channels, dtype=tf.float32),
                                            name=self.scope+'/biases')

    def __call__(self, inputs, nn_count, nn_index, filt_index, filt_coeff=None,
                 is_training=None):
        if filt_coeff is None:
            outputs = conv3d.conv3d(inputs, self.filter['depthwise'],
                                            nn_count, nn_index, filt_index)
        else:
            outputs = conv3d.fuzzyconv3d(inputs, self.filter['depthwise'],
                                                 nn_count, nn_index, filt_index, filt_coeff)

        outputs = tf.matmul(outputs, self.filter['pointwise'])
        outputs = tf.nn.bias_add(outputs, self.filter['biases'])
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs, is_training)

        return outputs


class PerItemConv3d(tf.Module):

    def __init__(self, in_channels, out_channels, scope='', use_xavier=True, stddev=1e-3,
                 with_bn=True, activation_fn=tf.nn.relu, bn_momentum=0.9):
        super(PerItemConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scope = scope
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        self.filter = {}
        if with_bn:
            self.BatchNorm = BatchNorm(bn_momentum=bn_momentum, scope=scope)
        self.reset_parameters()

    def reset_parameters(self):
        # pointwise kernel shape
        kernel_shape = [self.in_channels, self.out_channels]
        self.filter['pointwise'] = _filter_variable_(self.scope+'/pointwise_weights',
                                       shape=kernel_shape, use_xavier=self.use_xavier,
                                       stddev=self.stddev)
        # biases term
        self.filter['biases'] = tf.Variable(tf.zeros(self.out_channels, dtype=tf.float32),
                                            name=self.scope+'/biases')

    def __call__(self, inputs, is_training=None):
        outputs = tf.matmul(inputs, self.filter['pointwise'])
        outputs = tf.nn.bias_add(outputs, self.filter['biases'])
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs, is_training)
        return outputs


class Pool3d(tf.Module):
    """ 3D point cloud pooling.
    Args:
        pooling method, default to 'max'
    Returns:
        Variable tensor of shape [concat_Mp,C]
    """
    def __init__(self, method='max'):
        super(Pool3d, self).__init__()
        self.method = method

    def __call__(self, inputs, nn_count, nn_index):
        if self.method == 'max':
            outputs, max_index = pool3d.maxPool(inputs, nn_count, nn_index)
        elif self.method == 'avg':
            outputs = pool3d.avgPool(inputs, nn_count, nn_index)
        else:
            raise ValueError("Unknow pooling method %s."%self.method)

        return outputs


class Unpool3d(tf.Module):
    """ 3D unpooling
    Returns:
        outputs: float32 tensor of shape [concat_Np,C], unpooled features
    """
    def __init__(self):
        super(Unpool3d, self).__init__()

    def __call__(self, inputs, weight, nn_index):
        outputs = unpool3d.interpolate(inputs, weight, nn_index)
        return outputs


