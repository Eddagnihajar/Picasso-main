import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
pool3d_module = tf.load_op_library(os.path.join(srcDir, 'source/pcloud_pool3d_so.so'))


def maxPool(input, nn_count, nn_index):
    return pool3d_module.max_pool3d(input, nn_count, nn_index)
@tf.RegisterGradient('MaxPool3d')
def _max_pool3d_grad(op, grad_output, grad_index):
    input = op.inputs[0]
    max_index = op.outputs[1]
    grad_input = pool3d_module.max_pool3d_grad(input, grad_output, max_index)
    return [grad_input, None, None]


def avgPool(input, nn_count, nn_index):
    return pool3d_module.avg_pool3d(input, nn_count, nn_index)
@tf.RegisterGradient('AvgPool3d')
def _avg_pool3d_grad(op, grad_output):
    input = op.inputs[0]
    nn_count = op.inputs[1]
    nn_index = op.inputs[2]
    grad_input = pool3d_module.avg_pool3d_grad(input, grad_output, nn_count, nn_index)
    return [grad_input, None, None]

