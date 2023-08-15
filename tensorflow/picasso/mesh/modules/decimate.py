import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
decimation_module = tf.load_op_library(os.path.join(srcDir, 'source/decimate_so.so'))


def mesh_decimation_(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvOut, useArea=True, wgtBnd=5, factor=2):
    '''
    input:
        vertexIn:   (batch_npoints, 3+) float32 array, concatenated points with/without features
        faceIn:     (batch_nfaces, 3) int32 array, concatenated triangular faces
        geometryIn: (batch_nfaces, 5) float32 array, geometrics of each face which is
                                      composed of [normal=[nx,ny,nz],intercept=d,area]
        nvIn:       (batch,) int32 vector, point/vertex number of each sample in the batch
        mfIn:       (batch,) int32 vector, face number of each sample in the batch
        nvOut:      (batch,) int32 vector, expected number of points/vertices to output of each sample
        wgtBnd:      float scalar, weight boundary quadric error, (>1) preserves boundary edges
    returns:
        vertexOut:   (batch_mpoints, 3) float32 array, concatenated points with/without features
        faceOut:     (batch_mfaces, 3) int32 array, concatenated triangular faces
        geometryOut: (batch_mfaces, 5) float32 array, geometrics of each output face which is
                                       composed of [normal=[nx,ny,nz],intercept=d,area]
        nvOut:       (batch,) int32 vector, point/vertex number of each sample in the batch
        mfOut:       (batch,) int32 vector, face number of each sample in the batch
        vtReplace:   (batch_npoints,) int32 array, negative values (remove minus '-') for vertex to be
                                      contracted in each cluster. zero for vertex no change because it
                                      forms a cluster by itself positive values recording the cluster
                                      size excluding the vertex itself
        vtMap:       (batch_npoints,) int32 array, contracted/degenerated vertices got mapping to -1,
                                      the valid vertices got mapping start from 0
    '''
    nv2Remove = nvIn - nvOut
    repIn = tf.zeros(shape=tf.shape(vertexIn[:,0]), dtype=tf.int32)
    mapIn = tf.range(tf.shape(repIn)[0], dtype=tf.int32)

    loop_vars = [nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn]
    Bshape, Vshape = nv2Remove.get_shape(), repIn.get_shape()
    shape_invariants = [Bshape, tf.TensorShape([None,3]), tf.TensorShape([None,3]),
                        tf.TensorShape([None,5]), Vshape, Vshape, Bshape, Bshape]

    def condition(nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn):
        return tf.reduce_any(tf.greater(nv2Remove, 0))

    def body(nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn):
        nvIn_cumsum = tf.cumsum(nvIn, exclusive=False)
        mfIn_cumsum = tf.cumsum(mfIn, exclusive=False)

        # maxRemove = tf.maximum(tf.minimum(nv2Remove, nvIn//factor),1)
        vertexOut, faceOut, isDegenerate, repOut, mapOut, \
        nvOut, mfOut = decimation_module.mesh_decimation(vertexIn, faceIn, geometryIn, \
                                                         nvIn_cumsum, mfIn_cumsum, nv2Remove, \
                                                         useArea, wgtBnd)
        faceIn = tf.gather_nd(faceOut, tf.where(tf.logical_not(isDegenerate)))
        vertexIn = tf.gather_nd(vertexOut, tf.where(mapOut>=0))
        geometryIn = compute_triangle_geometry_(vertexIn, faceIn)
        nv2Remove = nv2Remove - (nvIn - nvOut)
        repIn, mapIn = combine_clusters_(repIn, mapIn, repOut, mapOut)
        nvIn, mfIn = nvOut, mfOut
        return [nv2Remove, vertexIn, faceIn, geometryIn, repIn, mapIn, nvIn, mfIn]

    final = tf.while_loop(condition, body, loop_vars, shape_invariants)
    vertexOut, faceOut, geometryOut, repOut, mapOut, nvOut, mfOut = final[1:]

    return vertexOut, faceOut, geometryOut, nvOut, mfOut, repOut, mapOut
tf.no_gradient('MeshDecimation')


def combine_clusters_(repA, mapA, repB, mapB):
    '''
       input:
            repA: (batch_points,) int32 array, vertex clustering information of LARGE input
            mapA: (batch_points,) int32 array, vertex mappinging information of LARGE input
            repB: (batch_points,) int32 array, vertex clustering information of SMALL/decimated input
            mapB: (batch_points,) int32 array, vertex mappinging information of SMALL/decimated input
       returns:
            repComb: (batch_points,) int32 array, vertex clustering information after merging LARGE/SMALL input
            mapComb: (batch_points,) int32 array, vertex mappinging information after merging LARGE/SMALL input
    '''
    repComb, mapComb = decimation_module.combine_clusters(repA, mapA, repB, mapB)
    return repComb, mapComb
tf.no_gradient('CombineClusters')


def compute_triangle_geometry_(vertex, face):
    vertex = vertex[:,:3]
    vec10 = tf.gather_nd(vertex,tf.expand_dims(face[:,1],axis=1)) - tf.gather_nd(vertex,tf.expand_dims(face[:,0],axis=1))
    vec20 = tf.gather_nd(vertex,tf.expand_dims(face[:,2],axis=1)) - tf.gather_nd(vertex,tf.expand_dims(face[:,0],axis=1))
    raw_normal = tf.linalg.cross(vec10, vec20)
    l2norm = tf.sqrt(tf.reduce_sum(tf.square(raw_normal), axis=1, keepdims=True))
    area = tf.divide(l2norm,2)
    normal = tf.math.divide_no_nan(raw_normal,l2norm) # unit length normal vectors
    v1 = tf.gather_nd(vertex,tf.expand_dims(face[:,0],axis=1))
    d = -tf.reduce_sum(tf.multiply(normal,v1),axis=1,keepdims=True)  # intercept of the triangle plane
    geometry = tf.concat([normal,d,area],axis=1)
    return geometry
tf.no_gradient('ComputeTriangleGeometry')


def count_vertex_adjface_(face, vtMap, vertexOut):
    '''
        Count the number of adjacent faces for output vertices, i.e. {vtMap[i] | vtMap[i]>=0},
    '''
    nfCount = decimation_module.count_vertex_adjface(face, vtMap, vertexOut[:,:3])
    return nfCount
tf.no_gradient('CountVertexAdjface')



