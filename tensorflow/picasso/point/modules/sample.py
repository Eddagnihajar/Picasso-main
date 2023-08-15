import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
module = tf.load_op_library(os.path.join(srcDir, 'source/sample_so.so'))
# print(dir(module))


def fps_(xyz_in, nv_in, nv_out):
    '''
    input:
        database:   (Np, 3) float32 array, database points
        nvDatabase: (batch_size) int32 vector, number of points of each batch sample
        nv_out: (batch_size) int32, number of output points of each batch sample
    returns:
        sample_index: (Mp) int32 array, index of sampled neurons in the database
    '''
    nv_in = tf.cumsum(nv_in, exclusive=False)
    nv_out = tf.cumsum(nv_out, exclusive=False)
    print(nv_in, nv_out)
    return module.farthest_point_sample(xyz_in, nv_in, nv_out)
tf.no_gradient('FarthestPointSample')



def fps_dim3_(database, mPoint):
    '''
    input:
        neursize: int32, the number of neurons/points to be sampled
        database: (batch, npoint, 3) float32 array, database points
    returns:
        neuron_index: (batch_size, neursize) int32 array, index of sampled neurons in the database
    '''
    return module.farthest_point_sample3d(database, mPoint)
tf.no_gradient('FarthestPointSample3D')

