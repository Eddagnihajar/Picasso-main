import tensorflow as tf
import tensorflow_addons as tfa
import os, sys
import numpy as np

from .modules.buildkernel import SPH3Dkernel_, fuzzySPH3Dkernel_
from .modules.nnquery import range_search_, cube_search_, knn3_search_
from .modules.sample import fps_


def range_search(xyz_data, xyz_query, nv_data, nv_query, radius, max_nn):
    cnt_info, nn_cnt, nn_idx, nn_dst = range_search_(xyz_data, xyz_query, nv_data, nv_query,
                                                     radius=radius, nnsample=max_nn)
    return nn_cnt, nn_idx, nn_dst


def cube_search(xyz_data, xyz_query, nv_data, nv_query, length, nnsample=None, gridsize=4):
    cnt_info, nn_cnt, nn_idx, nn_dst = cube_search_(xyz_data, xyz_query, nv_data, nv_query,
                                                    length=length, nnsample=nnsample, gridsize=gridsize)
    return nn_cnt, nn_idx, nn_dst


def knn3_search(xyz_data, xyz_query, nv_data, nv_query):
    nn_idx, nn_dst = knn3_search_(xyz_data, xyz_query, nv_data, nv_query)
    return nn_idx, nn_dst


def compute_unpool_weight(nn_dist):
    # default inverse distance based weights
    weight = 1/(nn_dist+np.finfo(float).eps)
    weight = weight/tf.reduce_sum(weight, axis=-1, keepdims=True)
    return weight


def fps(xyz_in, nv_xyz, num_sample):
    sample_index = fps_(xyz_in, nv_xyz, num_sample) # single dimension
    return sample_index


def build_range_graph(xyz_data, xyz_query, nv_data, nv_query,
                      radius, nn_uplimit=None, kernel_shape=None, fuzzy=True):
    nn_cnt, nn_idx, nn_dst = range_search(xyz_data, xyz_query, nv_data, nv_query,
                                          radius=radius, max_nn=nn_uplimit)
    if fuzzy:
        filt_idx, filt_coeff = fuzzySPH3Dkernel_(xyz_data, xyz_query, nn_idx, nn_dst, radius,
                                                 kernel=kernel_shape)
    else:
        filt_idx = SPH3Dkernel_(xyz_data, xyz_query, nn_idx, nn_dst,
                                radius, kernel=kernel_shape)
        filt_coeff = None
    return nn_cnt, nn_idx, nn_dst, filt_idx, filt_coeff


def build_cube_graph(xyz_data, xyz_query, nv_data, nv_query,
                     radius, nn_uplimit=None, grid_size=None):
    cnt_info, nn_cnt, nn_idx = cube_search(xyz_data, xyz_query, nv_data, nv_query, length=2*radius,
                                           nnsample=nn_uplimit, gridsize=grid_size)
    filt_idx = nn_idx[:, -1]
    nn_idx = nn_idx[:, :-1]
    nn_dst, filt_coeff = None, None
    return nn_cnt, nn_idx, nn_dst, filt_idx, filt_coeff
