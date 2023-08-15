import tensorflow as tf
import os, sys
import numpy as np
from typing import List
from termcolor import colored
from .modules.decimate import mesh_decimation_, compute_triangle_geometry_, \
                              count_vertex_adjface_, combine_clusters_


def mesh_decimation(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvSample, useArea=True, wgtBnd=1.):
    vertexOut, faceOut, geometryOut, nvOut, mfOut,  \
    repOut, mapOut = mesh_decimation_(vertexIn, faceIn, geometryIn, nvIn, mfIn, nvSample,
                                      useArea=useArea, wgtBnd=wgtBnd)
    return vertexOut, faceOut, geometryOut, nvOut, mfOut, repOut, mapOut


def compute_triangle_geometry(vertex, face):
    geometry = compute_triangle_geometry_(vertex, face)
    return geometry


def count_vertex_adjface(face, vtMap, vertexOut):
    nf_count = count_vertex_adjface_(face, vtMap, vertexOut)
    return nf_count


def combine_vertex_clusters(rep_highRes, map_highRes, rep_lowRes, map_lowRes):
    rep_combined, map_combined = combine_clusters_(rep_highRes, map_highRes,
                                                  rep_lowRes, map_lowRes)
    return rep_combined, map_combined


class MeshHierarchy:
    def __init__(self, *params):
        self.num_params = len(params)
        if self.num_params==3:
            self.nv_samples, self.useArea, self.wgtBnd = params
            self.Iters = len(self.nv_samples)
        elif self.num_params==4:
            self.stride, self.min_nvOut, self.useArea, self.wgtBnd = params
            self.Iters = len(self.stride)

    def __call__(self, vertex_in, face_in, nv_in, mf_in):
        mesh_hierarchy = []
        geometry_in = compute_triangle_geometry(vertex_in, face_in)
        mesh_hierarchy.append((vertex_in, face_in, geometry_in, nv_in, mf_in, None, None))
        for l in range(self.Iters):
            if self.num_params==3:
                nv_out = self.nv_samples[l]
            elif self.num_params==4:
                nv_out = tf.cast(tf.cast(nv_in,tf.float32)/float(self.stride[l]),tf.int32)
                nv_out = tf.maximum(nv_out, self.min_nvOut[l])
            dec_mesh = mesh_decimation(vertex_in, face_in, geometry_in, nv_in, mf_in,
                                       nv_out, self.useArea, self.wgtBnd)
            mesh_hierarchy.append(dec_mesh)
            vertex_in, face_in, geometry_in, nv_in, mf_in = dec_mesh[:5]
        return mesh_hierarchy


def crop_mesh(vertex, face, min_xyz, max_xyz, vertex_labels=None, facet_labels=None, crop_hgt=False):
    if crop_hgt:
        crop_vertex_mask = (vertex[:,0]>=min_xyz[0]) & (vertex[:,0]<=max_xyz[0]) &\
                           (vertex[:,1]>=min_xyz[1]) & (vertex[:,1]<=max_xyz[1]) &\
                           (vertex[:,2]>=min_xyz[2]) & (vertex[:,2]<=max_xyz[2])
    else:
        crop_vertex_mask = (vertex[:,0]>=min_xyz[0]) & (vertex[:,0]<=max_xyz[0]) &\
                           (vertex[:,1]>=min_xyz[1]) & (vertex[:,1]<=max_xyz[1])
    crop_face_mask = (tf.gather(crop_vertex_mask, face[:,0])) & \
                     (tf.gather(crop_vertex_mask, face[:,1])) & \
                     (tf.gather(crop_vertex_mask, face[:,2]))
    crop_face_indices = tf.squeeze(tf.where(crop_face_mask))
    crop_face = tf.gather(face, crop_face_indices)
    crop_xyz_indices, idx = tf.unique(tf.reshape(crop_face, [-1]))
    num_vertices = crop_xyz_indices.shape[0]

    if num_vertices==0:
        return None

    crop_face = tf.reshape(idx, [-1,3])
    crop_vertex = tf.gather(vertex, crop_xyz_indices)
    crop_labels = None
    if vertex_labels is not None:
        crop_labels = tf.gather(vertex_labels, crop_xyz_indices)
    if facet_labels is not None:
        crop_labels = tf.gather(facet_labels, crop_face_indices)
    return crop_vertex, crop_face, crop_labels # tuple


def voxelize_mesh(vertex, face, voxel_size, seg_labels=None, return_inverse=False):
    assert (voxel_size>0, "")
    # print (colored("We assume the vertex size of per-mesh is less than 1 billion.","red"))

    labels = tf.cast(seg_labels[:,tf.newaxis],dtype=tf.float32)
    feature = tf.concat([vertex,labels],axis=1)

    pos_xyz = vertex[:,:3]-tf.reduce_min(vertex[:,:3],axis=0)
    tmp_ids = tf.math.floor(pos_xyz/(voxel_size/100)) # centimeter voxel_size to meter
    Mags = tf.math.ceil(tf.math.log(tf.reduce_max(tmp_ids,axis=0))/tf.math.log(10.))

    ids = tf.cast(tmp_ids, dtype=tf.int32)
    Mags = tf.cast(Mags, dtype=tf.int32)

    ids_1D = ids[:,0]*(10**(Mags[1]+Mags[0])) + ids[:,1]*(10**Mags[0]) + ids[:,2]
    id_unique, idx = tf.unique(ids_1D)
    num_segments = tf.shape(id_unique)[0]

    mean = tf.math.unsorted_segment_mean(feature, idx, num_segments)
    new_vertex = mean[:,:-1]
    voxelized_labels = mean[:,-1:]

    # get labels, and face ids
    sorted_labels = tf.gather(voxelized_labels,idx)
    residual = tf.abs(sorted_labels - labels)
    edge_voxel_indices, _ = tf.unique(tf.gather(idx, tf.where(residual[:,0]>1e-4)[:,0]))
    indicator = tf.scatter_nd(edge_voxel_indices[:,tf.newaxis],
                              -tf.ones_like(edge_voxel_indices),
                              shape=[voxelized_labels.shape[0]])
    new_labels = tf.where(indicator==-1, indicator,
                          tf.cast(voxelized_labels[:,0], dtype=tf.int32))

    V1 = tf.gather(idx, face[:,0])
    V2 = tf.gather(idx, face[:,1])
    V3 = tf.gather(idx, face[:,2])
    non_degenerate = tf.logical_not((V1==V2) | (V2==V3) | (V3==V1))
    voxelized_face = tf.stack([V1,V2,V3],axis=1)
    new_face = tf.gather(voxelized_face, tf.where(non_degenerate)[:,0])

    if return_inverse:
        return new_vertex, new_face, new_labels, idx
    else:
        return new_vertex, new_face, new_labels


