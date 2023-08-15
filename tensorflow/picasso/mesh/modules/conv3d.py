import os
import tensorflow as tf
srcDir = os.path.dirname(os.path.abspath(__file__))
module = tf.load_op_library(os.path.join(srcDir, "C:/Users/exe/Desktop/Picasso-main/tensorflow/picasso/mesh/modules/source/mesh_conv3d.cpp"))


def facet2facet(input, filter, baryCoeff, numInterior):
    '''
    Compute the feature of each triangular face based on the provided interpolated interior features.
    The common input feature for this function is facet textures.
    Input:
        input:       (concat_ptNiK, in_channels) float32 array, input facet interpolated features
        filter:      (out_channels, in_channels, 3) float32 array, convolution filter
        baryCoeff:   (concat_ptNiK, 3) float32 array, face interior interpolation weights,
                                                      Barycentric Coordinates
        numInterior: (concat_NfIn) int32 vector, number of interpolated interior points in each facet
    Output:
        output: (concat_NfIn, out_channels) float32 array, output facet features
                                            out_channels = in_channels*multiplier
    '''
    return module.facet2facet_conv3d(input, filter, baryCoeff, numInterior)
@tf.RegisterGradient('Facet2facetConv3d')
def _facet2facet_conv3d_grad(op, grad_output):
    input  = op.inputs[0]
    filter = op.inputs[1]
    baryCoeff  = op.inputs[2]
    numInterior = op.inputs[3]
    grad_input, grad_filter = module.facet2facet_conv3d_grad(input, filter, grad_output,
                                                             baryCoeff, numInterior)
    return [grad_input, grad_filter, None, None]


def vertex2facet(input, filter, face):
    '''
    Compute the feature of each triangular face based on its vertices' features by interpolation
    Input:
        input:       (concat_NvIn, in_channels) float32 array, input vertex/point features
        filter:      (3, in_channels, multiplier) float32 array, convolution filter
        face:        (concat_NfIn, 3) int32 array, vertex list of each facet
    Output:
        output:      (concat_NfIn, out_channels) float32 array, output facet features
                                                 out_channels = in_channels * multiplier
    '''
    return module.vertex2facet_conv3d(input, filter, face)
@tf.RegisterGradient('Vertex2facetConv3d')
def _vertex2facet_conv3d_grad(op, grad_output):
    input  = op.inputs[0]
    filter = op.inputs[1]
    face = op.inputs[2]
    grad_input, grad_filter = module.vertex2facet_conv3d_grad(input, filter, grad_output, face)
    return [grad_input, grad_filter, None]


def facet2vertex(input, filter, coeff, face, nfCount, vtMap):
    '''
        Input:
            input:   (concat_NfIn, in_channels) float32 array, input facet features
            filter:  (modelSize, in_channels, multiplier) float32 array, convolution filter
            coeff:   (concat_NfIn, modelSize) float32 array, coefficients for each model
            face:    (concat_NfIn, 3) int32 array, vertex list of each facet
            nfCount: (concat_NvIn) int32 vector, number of adjacent faces of each vertex
            vtMap:   (concat_NvIn) int32 vector, input to output vertex index mapping
        Output:
            output:  (concat_NvOut, out_channels) float32 array, output vertex/point features,
                                                  out_channels = in_channels * multiplier
        '''
    return module.facet2vertex_conv3d(input, filter, coeff, face, nfCount, vtMap)
@tf.RegisterGradient('Facet2vertexConv3d')
def _facet2vertex_conv3d_grad(op, grad_output):
    input = op.inputs[0]
    filter = op.inputs[1]
    coeff = op.inputs[2]
    face = op.inputs[3]
    nfCount = op.inputs[4]
    vtMap = op.inputs[5]
    grad_input, grad_filter, grad_coeff = module.facet2vertex_conv3d_grad(input, filter, coeff, grad_output,
                                                       face, nfCount, vtMap)
    # grad_coeff = None
    return [grad_input, grad_filter, grad_coeff, None, None, None]

