import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
buildkernel_module = tf.load_op_library(os.path.join(srcDir, 'source/buildkernel_so.so'))


def SPH3Dkernel_(database, query, nn_index, nn_dist, radius, kernel=[8,2,3]):
    '''
    Input:
        database: (concat_Np, 3+) float32 array, database points (x,y,z,...)
        query:    (concat_Mp, 3+) float32 array, query points (x,y,z,...)
        nn_index: (Nout, 2) int32 array, neighbor indices
        nn_dist:  (Nout) float32, sqrt distance array
        radius:   float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (Nout) int32 array, filter bin indices
    '''
    n, p, q = kernel

    database = database[:, 0:3]  #(x,y,z)
    query = query[:, 0:3] #(x,y,z)
    return buildkernel_module.spherical_kernel(database, query, nn_index, nn_dist, radius, n, p, q)
tf.no_gradient('SphericalKernel')


def fuzzySPH3Dkernel_(database, query, nn_index, nn_dist, radius, kernel=[8,4,1]):
    '''
    Input:
        database: (concat_Np, 3+) float32 array, database points (x,y,z,...)
        query:    (concat_Mp, 3+) float32 array, query points (x,y,z,...)
        nn_index: (Nout, 2) int32 array, neighbor indices
        nn_dist:  (Nout) float32, sqrt distance array
        radius:   float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (Nout, 3) int32 array, fuzzy filter indices,
                    (8=2*2*2, 2 for each splitting dimension)
        filt_coeff: (Nout, 3) float32 array, fuzzy filter weights,
                    kernelsize=prod(kernel)+1
    '''
    n, p, q = kernel

    database = database[:, 0:3]  #(x,y,z)
    query = query[:, 0:3] #(x,y,z)
    return buildkernel_module.fuzzy_spherical_kernel(database, query, nn_index, nn_dist, radius, n, p, q)
tf.no_gradient('FuzzySphericalKernel')







