import tensorflow as tf
from typing import List
import picasso.mesh.utils as meshUtil
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
    var = tf.Variable(initial_value=initializer(shape=shape), name=name, dtype=tf.float32)
    return var


class F2FConv3d(tf.Module):
    """ convolution on the facet textures
    Args:
       in_channels:  number of input  channels
       out_channels: number of output channels
       use_xavier: bool, use xavier_initializer if true
       stddev: float, stddev for truncated_normal init
    Returns:
       Variable tensor
    """
    def __init__(self, in_channels, out_channels, scope='', use_xavier=True, stddev=1e-3,
                 with_bn=True, activation_fn=tf.nn.relu, bn_momentum=0.9):
        super(F2FConv3d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scope = scope
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        self.filter = {}
        self.reset_parameters()
        if self.with_bn:
            self.BatchNorm = BatchNorm(bn_momentum=bn_momentum, scope=scope)

    def reset_parameters(self):
        # depthwise kernel shape
        f2f_kernel_shape = [self.out_channels, self.in_channels, 3]
        self.filter['weights'] = _filter_variable_(self.scope+'/F2F_weights',
                                       shape=f2f_kernel_shape, use_xavier=self.use_xavier,
                                       stddev=self.stddev)

        # biases term
        self.filter['biases'] = tf.Variable(tf.zeros(self.out_channels,dtype=tf.float32),
                                            name=self.scope+'/F2F_biases')

    def __call__(self, input_texture, bary_coeff, num_texture, is_training=None):
        num_texture = tf.cumsum(num_texture,)  # cumulative sum required
        outputs = conv3d.facet2facet(input_texture, self.filter['weights'],
                                     bary_coeff, num_texture)
        outputs = tf.nn.bias_add(outputs, self.filter['biases'])
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs, is_training)
        return outputs


class V2FConv3d(tf.Module):
    """ feature propagation from vertex to facet
    Args:
        in_channels:  number of input  channels
        out_channels: number of output channels
        depth_multiplier: depth multiplier in the separable convolution
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
    Returns:
        Variable tensor
    """
    def __init__(self, in_channels, out_channels, depth_multiplier=1,
                 scope='', use_xavier=True, stddev=1e-3, with_bn=True,
                 activation_fn=tf.nn.relu, bn_momentum=0.9):
        super(V2FConv3d,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multiplier = depth_multiplier
        self.scope = scope
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        self.filter = {}
        self.reset_parameters()
        if self.with_bn:
            self.BatchNorm = BatchNorm(bn_momentum=bn_momentum, scope=scope)

    def reset_parameters(self):
        # depthwise kernel shape
        v2f_kernel_shape = [3, self.in_channels, self.multiplier]
        self.filter['depthwise'] = _filter_variable_(self.scope+'/V2F_depthwise_weights',
                                      shape=v2f_kernel_shape, use_xavier=self.use_xavier,
                                      stddev=self.stddev)

        # pointwise kernel shape
        kernel_shape = [self.in_channels*self.multiplier, self.out_channels]
        self.filter['pointwise'] = _filter_variable_(self.scope+'/V2F_pointwise_weights',
                                      shape=kernel_shape, use_xavier=self.use_xavier,
                                      stddev=self.stddev)
        # biases term
        self.filter['biases'] = tf.Variable(tf.zeros(self.out_channels, dtype=tf.float32),
                                            name=self.scope+'/V2F_biases')

    def __call__(self, inputs, face, is_training=None):
        outputs = conv3d.vertex2facet(inputs, self.filter['depthwise'], face)
        outputs = tf.matmul(outputs, self.filter['pointwise'])
        outputs = tf.nn.bias_add(outputs, self.filter['biases'])
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs, is_training)
        return outputs


class F2VConv3d(tf.Module):
    """ convolution to learn vertex features from adjacent facets
     Args:
         in_channels:  number of input  channels
         out_channels: number of output channels
         kernel_size:  number of components in the mixture model
         depth_multiplier: depth multiplier in the separable convolution
         use_xavier: bool, use xavier_initializer if true
         stddev: float, stddev for truncated_normal init
     Returns:
         Variable tensor
     """
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1,
                 scope='', use_xavier=True, stddev=1e-3, with_bn=True,
                 activation_fn=tf.nn.relu, bn_momentum=0.9):
        super(F2VConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.multiplier = depth_multiplier
        self.scope = scope
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.with_bn = with_bn
        self.activation_fn = activation_fn
        self.filter = {}
        self.reset_parameters()
        if self.with_bn:
            self.BatchNorm = BatchNorm(bn_momentum=bn_momentum, scope=scope)

    def reset_parameters(self):
        # depthwise kernel shape
        f2v_kernel_shape = [self.kernel_size, self.in_channels, self.multiplier]
        self.filter['depthwise'] = _filter_variable_(self.scope+'/F2V_depthwise_weights',
                                      shape=f2v_kernel_shape, use_xavier=self.use_xavier,
                                      stddev=self.stddev)
        # pointwise kernel shape
        kernel_shape = [self.in_channels*self.multiplier, self.out_channels]
        self.filter['pointwise'] = _filter_variable_(self.scope+'/F2V_pointwise_weights',
                                      shape=kernel_shape, use_xavier=self.use_xavier,
                                      stddev=self.stddev)
        # biases term
        self.filter['biases'] = tf.Variable(tf.zeros(self.out_channels,dtype=tf.float32),
                                                name=self.scope+'/F2V_biases')

    def __call__(self, inputs, face, nf_count, vt_map, filt_coeff, is_training=None):
        outputs = conv3d.facet2vertex(inputs, self.filter['depthwise'],
                                              filt_coeff, face, nf_count, vt_map)
        outputs = tf.matmul(outputs, self.filter['pointwise'])
        outputs = tf.nn.bias_add(outputs, self.filter['biases'])
        if self.activation_fn is not None:
            outputs = self.activation_fn(outputs)
        if self.with_bn:
            outputs = self.BatchNorm(outputs, is_training)
        return outputs


class V2VConv3d(tf.Module):
    # out_channels and depth_multiplier are vectors of 2 values.
    def __init__(self, in_channels, dual_out_channels, kernel_size,
                 dual_depth_multiplier:List[int], scope='', use_xavier=True, stddev=1e-3,
                 with_bn=True, activation_fn=tf.nn.relu, bn_momentum=0.9):
        super(V2VConv3d, self).__init__()
        self.V2F_conv3d = V2FConv3d(in_channels, dual_out_channels[0],
                                    dual_depth_multiplier[0], scope,
                                    use_xavier, stddev, with_bn,
                                    activation_fn, bn_momentum)
        self.F2V_conv3d = F2VConv3d(dual_out_channels[0], dual_out_channels[1],
                                    kernel_size, dual_depth_multiplier[1],
                                    scope, use_xavier, stddev, with_bn,
                                    activation_fn, bn_momentum)

    def __call__(self, inputs, face, nf_count, vt_map, filt_coeff, is_training=None):
        outputs = self.V2F_conv3d(inputs, face, is_training)
        outputs = self.F2V_conv3d(outputs, face, nf_count, vt_map, filt_coeff, is_training)
        return outputs


class PerItemConv3d(tf.Module):
    """ elementwise convolutionwith non-linear operation.
        e.g. per-pixel, per-point, per-voxel, per-facet, etc.
    Args:
        in_channels:  number of input  channels
        out_channels: number of output channels
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
    Returns:
      Variable tensor
    """
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
        self.reset_parameters()
        if with_bn:
            self.BatchNorm = BatchNorm(bn_momentum=bn_momentum, scope=scope)

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


class GlobalPool3d(tf.Module):
    """ 3D Global Mesh pooling.
    """
    def __init__(self, method='avg', scope=''):
        super(GlobalPool3d, self).__init__()
        self.method = method
        self.scope = scope

    def __call__(self, inputs, nv_in):
        batch_size = tf.shape(nv_in)[0]
        temp_nv_in = tf.cumsum(nv_in, exclusive=False)
        temp_nv_in = tf.concat([[0], temp_nv_in], axis=0)
        ta = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        for b in tf.range(batch_size):
            start = temp_nv_in[b]
            limit = temp_nv_in[b+1]
            indices = tf.range(start, limit, dtype=tf.int32)
            feats_instance = tf.gather(inputs, indices)  # shape: 1xin_channels
            if self.method == 'max':
                global_features = tf.reduce_max(feats_instance, axis=0, keepdims=True)
            elif self.method == 'avg':
                global_features = tf.reduce_mean(feats_instance, axis=0, keepdims=True)
            else:
                raise ValueError("Unknow pooling method %s."%self.method)
            ta = ta.write(b, global_features)
        outputs = tf.reshape(ta.concat(), shape=[batch_size, inputs.shape[-1]])
        return outputs


class Pool3d(tf.Module):
    """ 3D Mesh pooling.
    Args:
        pooling method, default to 'max'
    Returns:
        Variable tensor
    """
    def __init__(self, method='max'):
        super(Pool3d, self).__init__()
        self.method = method

    def __call__(self, inputs, vt_replace, vt_map, vt_out, weight=None):
        if self.method == 'max':
            outputs, max_index = pool3d.maxPool(inputs, vt_replace, vt_map, vt_out)
        elif self.method == 'avg':
            outputs = pool3d.avgPool(inputs, vt_replace, vt_map, vt_out)
        else:
            raise ValueError("Unknow pooling method %s."%self.method)

        return outputs


class Unpool3d(tf.Module):
    """ 3D Mesh unpooling
    Returns:
        Variable tensor
    """
    def __init__(self):
        super(Unpool3d, self).__init__()

    def __call__(self, inputs, vt_replace, vt_map):
        outputs = unpool3d.interpolate(inputs, vt_replace, vt_map)
        return outputs


class BatchNorm(tf.Module):
    def __init__(self, bn_momentum=0.9, scope=''):
        super(BatchNorm,self).__init__()
        self.norm_fn = tf.keras.layers.BatchNormalization(axis=-1, momentum=bn_momentum,
                                        beta_regularizer=tf.keras.regularizers.l2(1.0),
                                        gamma_regularizer=tf.keras.regularizers.l2(1.0),
                                        name=scope+'/BN')

    def __call__(self, data, is_training):
        return self.norm_fn(data, training=is_training)


class Dropout(tf.Module):
    def __init__(self, drop_rate=0.6):
        super(Dropout,self).__init__()
        self.DP = tf.keras.layers.Dropout(rate=drop_rate)

    def __call__(self, inputs, is_training):
        return self.DP(inputs, training=is_training)


class BuildF2VCoeff(tf.Module):
    ''' Compute fuzzy coefficients for the facet2vertex convolution
        Returns:
            fuzzy filter coefficients
        '''
    def __init__(self, kernel_size, tau=0.1, dim=3, stddev=1., scope=""):
        super(BuildF2VCoeff, self).__init__()
        self.kernel_size = kernel_size
        self.scope = scope
        self.centers = None
        self.tau = tau # similar to the temperature parameter in ContrastLoss
        self.dim = dim
        self.stddev = stddev
        self.reset_parameters()

    def reset_parameters(self):
        initializer = tf.keras.initializers.RandomNormal(stddev=self.stddev)
        shape = [self.kernel_size, self.dim]
        init_centers = initializer(shape=shape)

        # the centers is trainable
        self.centers = tf.Variable(initial_value=init_centers, trainable=True,
                                   name=self.scope+'/centers', dtype=tf.float32)

    def __call__(self, face_normals):
        # Note: face normal features are already unit vectors
        # normalize clustering center of normals to unit vectors
        l2norm = tf.sqrt(tf.reduce_sum(tf.square(self.centers), axis=-1, keepdims=True))
        norm_center = tf.math.divide(self.centers, l2norm)  # unit length normal vectors

        cos_theta = tf.matmul(face_normals, norm_center, transpose_b=True)  # cos(theta): range [-1,1]
        zeta = cos_theta/self.tau
        coeff = tf.exp(zeta)
        coeff = tf.divide(coeff, tf.reduce_sum(coeff, axis=-1, keepdims=True)) # softmax compute
        return coeff


