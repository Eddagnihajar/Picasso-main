import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
conv3d_module = tf.load_op_library(os.path.join(srcDir, 'source/pcloud_conv3d_so.so'))


def conv3d(input, filter, nn_count, nn_index, bin_index):
    '''
    Input:
        input:     (concat_Np, in_channels) float32 array, input point features
        filter:    (binsize, in_channels, channel_multiplier) float32 array, convolution filter
        nn_count:  (concat_Mp) int32 array, number of neighbors
        nn_index:  (Nout, 2) int32 array, neighbor indices
        bin_index: (Nout), filtet bins' indices
    Output:
        output: (concat_Mp, out_channels) float32 array, output point features
    '''
    return conv3d_module.depthwise_conv3d(input, filter, nn_count, nn_index, bin_index)
@tf.RegisterGradient("DepthwiseConv3d")
def _depthwise_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    nn_count = op.inputs[2]
    nn_index = op.inputs[3]
    bin_index = op.inputs[4]
    grad_input, grad_filter = conv3d_module.depthwise_conv3d_grad(input, filter, grad_output,
                                                                  nn_count, nn_index, bin_index)
    return [grad_input, grad_filter, None, None, None]


def fuzzyconv3d(input, filter, nn_count, nn_index, bin_index, bin_coeff):
    '''
    Input:
        input:     (concat_Np, in_channels) float32 array, input point features
        filter:    (binsize, in_channels, channel_multiplier) float32 array, convolution filter
        nn_count:  (concat_Mp) int32 array, number of neighbors
        nn_index:  (Nout, 2) int32 array, neighbor indices
        bin_index: (Nout, 3) int32 array, filtet bins' indices
        bin_coeff: (Nout, 3) float32 array, kernel bin coefficients
    Output:
        output:    (concat_Mp, out_channels) float32 array, output point features
    '''
    return conv3d_module.fuzzy_depthwise_conv3d(input, filter, nn_count, nn_index, bin_index, bin_coeff)
@tf.RegisterGradient("FuzzyDepthwiseConv3d")
def _fuzzy_depthwise_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    nn_count = op.inputs[2]
    nn_index = op.inputs[3]
    bin_index = op.inputs[4]
    bin_coeff = op.inputs[5]
    grad_input, grad_filter = conv3d_module.fuzzy_depthwise_conv3d_grad(input, filter, grad_output,
                                                                        nn_count, nn_index,
                                                                        bin_index, bin_coeff)
    return [grad_input, grad_filter, None, None, None, None]



def conv3d_fast(input, filter, nn_index, nn_count, bin_index):
    '''
    Input:
        input:   (batch, npoint, in_channels) float32 array, input point features
        filter: (binsize, in_channels, out_channels) float32 array, convolution filter
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        bin_index: (batch, mpoint, nnsample), filtet bins' indices
    Output:
        output: (batch, mpoint, out_channels) float32 array, output point features
    '''
    batch_size, mpoint, num_sample = nn_index.get_shape().as_list()
    bin_size, in_channels, out_channels = filter.get_shape().as_list()
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, mpoint, num_sample, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(nn_index, axis=3)], axis=3)

    input_padded = tf.gather_nd(input, indices) # (batch, mpoint, nnsample, in_channels)
    filter_onehot = tf.one_hot(bin_index, depth=bin_size, on_value=1.0,
                               off_value=0.0, dtype=tf.float32) # (batch, mpoint, nnsample, binsize)
    filter_onehot = tf.transpose(filter_onehot, [0, 1, 3, 2]) # (batch, mpoint, binsize, nnsample)

    selected_input_padded = tf.matmul(filter_onehot, input_padded) # (batch, mpoint, binsize, in_channels)
    selected_input_padded = tf.reshape(selected_input_padded, (-1, bin_size, in_channels))
    selected_input_padded = tf.transpose(selected_input_padded, [1, 0, 2]) # (binsize, batch, mpoint, in_channels)

    output = tf.matmul(selected_input_padded, filter) # (bin_size, batch_size*mpoint, out_channels)
    output = tf.reduce_sum(output, axis=0) # (batch_size*mpoint, out_channels)
    output = tf.reshape(output, (batch_size, mpoint, -1)) # (batch_size, mpoint, out_channels)

    nn_count = tf.cast(nn_count, tf.float32)
    nn_count = tf.tile(tf.reshape(nn_count,(batch_size,mpoint,1)), (1, 1, out_channels))
    output = tf.divide(output, nn_count)

    return output
