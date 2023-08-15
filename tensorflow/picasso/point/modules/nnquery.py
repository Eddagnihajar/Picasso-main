import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
nnquery_module = tf.load_op_library(os.path.join(srcDir, 'source/nnquery_so.so'))


def range_search_(database, query, nvDatabase, nvQuery,
                  radius=0.1, nnsample=None):
    '''
    Input:
        database: (concat_Np, 3+x) float32 array, database points
        query:    (concat_Mp, 3) float32 array, query points
        radius:   float32, range search radius
        dilation_rate: float32, dilation rate of range search
        nnsample: int32, maximum number of neighbors to be sampled
    Output:
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor and filter bin indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        nn_dist(optional): (batch, mpoint, nnsample) float32, sqrt distance array
    '''
    database = database[...,0:3]
    query = query[...,0:3]
    nvDatabase = tf.cumsum(nvDatabase, exclusive=False)
    nvQuery = tf.cumsum(nvQuery, exclusive=False)

    if nnsample is None:
        nnsample = 2147483647 # int32 maximum value
    # print('nnSample:', nnsample)

    cntInfo, nnIndex, nnDist = nnquery_module.build_sphere_neighbor(database, query, nvDatabase,
                                                                    nvQuery, radius, nnsample)
    nnCount = tf.concat([cntInfo[:1],cntInfo[1:]-cntInfo[:-1]],axis=0)
    nnCount = tf.cast(nnCount, dtype=tf.int32)
    return cntInfo, nnCount, nnIndex, nnDist
tf.no_gradient('BuildSphereNeighbor')



def cube_search_(database, query, nvDatabase, nvQuery,
                 length=0.1, dilation_rate=None, nnsample=None, gridsize=3):
    '''
    Input:
        database: (concat_Np, 3) float32 array, database points
        query:    (concat_Mp, 3) float32 array, query points
        length:   float32, cube search length
        dilation_rate: float32, dilation rate of cube search
        nnsample: int32, maximum number of neighbors to be sampled
        gridsize: int32 , cubical kernel size
    Output:
        nn_index: (batch, mpoint, nnsample, 2) int32 array, neighbor and filter bin indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
    '''

    database = database[..., 0:3]
    query = query[..., 0:3]
    nvDatabase = tf.cumsum(nvDatabase, exclusive=False)
    nvQuery = tf.cumsum(nvQuery, exclusive=False)

    if dilation_rate is not None:
        length = dilation_rate * length

    if nnsample is None:
        nnsample = 2147483647 # int32 maximum value
    # print('nnSample:', nnsample)

    cntInfo, nnIndex = nnquery_module.build_cube_neighbor(database, query, nvDatabase, nvQuery,
                                                          length, nnsample, gridsize)
    nnCount = tf.concat([cntInfo[:1], cntInfo[1:] - cntInfo[:-1]], axis=0)
    nnCount = tf.cast(nnCount, dtype=tf.int32)
    return cntInfo, nnCount, nnIndex
tf.no_gradient('BuildCubeNeighbor')



def knn3_search_(database, query, nvDatabase, nvQuery):
    '''
    Input:
        database: (batch, npoint, 3) float32 array, database points
        query:    (batch, mpoint, 3) float32 array, query points
    Output:
        nn_index: (batch, mpoint, 3) int32 array, neighbor indices
        nn_dist(optional): (batch, mpoint, 3) float32, sqrt distance array
    '''
    # Return the 3 nearest neighbors of each point
    database = database[..., 0:3]
    query = query[..., 0:3]
    nvDatabase = tf.cumsum(nvDatabase, exclusive=False)
    nvQuery = tf.cumsum(nvQuery, exclusive=False)

    nn_index, nn_dist = nnquery_module.build_nearest_neighbor(database, query, nvDatabase, nvQuery)
    return nn_index, nn_dist
tf.no_gradient('BuildNearestNeighbor')