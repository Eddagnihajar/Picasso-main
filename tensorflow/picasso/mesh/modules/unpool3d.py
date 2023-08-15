import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
unpool3d_module = tf.load_op_library(os.path.join(srcDir, 'source/mesh_unpool3d_so.so'))


def interpolate(input, vt_replace, vt_map):
    return unpool3d_module.mesh_interpolate(input, vt_replace, vt_map)
@tf.RegisterGradient("MeshInterpolate")
def _interpolate_grad(op, grad_output):
    input = op.inputs[0]
    vt_replace = op.inputs[1]
    vt_map = op.inputs[2]
    grad_input = unpool3d_module.mesh_interpolate_grad(input, grad_output, vt_replace, vt_map)
    return [grad_input, None, None]


