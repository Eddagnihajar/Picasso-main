import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
pool3d_module = tf.load_op_library(os.path.join(srcDir, 'source/mesh_pool3d_so.so'))


def maxPool(input, vt_replace, vt_map, vt_out):
    return pool3d_module.mesh_max_pool3d(input, vt_replace, vt_map, vt_out)
@tf.RegisterGradient('MeshMaxPool3d')
def _max_pool3d_grad(op, grad_output, grad_index):
    input = op.inputs[0]
    max_index = op.outputs[1]
    grad_input = pool3d_module.mesh_max_pool3d_grad(input, grad_output, max_index)
    return [grad_input, None, None, None]


def avgPool(input, vt_replace, vt_map, vt_out):
    '''
        During the decimation, we record the vertex clusters in vt_map.
        Here we perform average pooling to get feature of the output vertex
        which is contracted from the input vertex clusters.
    '''
    return pool3d_module.mesh_avg_pool3d(input, vt_replace, vt_map, vt_out)
@tf.RegisterGradient('MeshAvgPool3d')
def _avg_pool3d_grad(op, grad_output):
    input = op.inputs[0]
    vt_replace = op.inputs[1]
    vt_map = op.inputs[2]
    grad_input = pool3d_module.mesh_avg_pool3d_grad(input, grad_output, vt_replace, vt_map)
    return [grad_input, None, None, None]



