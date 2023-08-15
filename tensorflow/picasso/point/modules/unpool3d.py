import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
unpool3d_module = tf.load_op_library(os.path.join(srcDir, 'source/pcloud_unpool3d_so.so'))


def interpolate(input, weight, nn_index):
    return unpool3d_module.weighted_interpolate(input, weight, nn_index)
@tf.RegisterGradient("WeightedInterpolate")
def _weighted_interpolate_grad(op, grad_output):
    input = op.inputs[0]
    weight = op.inputs[1]
    nn_index = op.inputs[2]
    grad_input = unpool3d_module.weighted_interpolate_grad(input, grad_output, weight, nn_index)
    return [grad_input, None, None]
