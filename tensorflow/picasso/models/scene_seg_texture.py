import numpy as np
import open3d as o3d
from typing import List
import math
from termcolor import colored
import tensorflow as tf
import picasso.mesh.utils as meshUtil
from picasso.mesh.layers import V2FConv3d, PerItemConv3d, BuildF2VCoeff
from picasso.mesh.layers import Pool3d as mesh_pool
from picasso.mesh.layers import Unpool3d as mesh_unpool
from .pi2_blocks import DualBlock, EncoderMeshBlock
from .pi2_blocks import FirstBlockTexture as FirstBlock


class PicassoNetII(tf.Module):
    def __init__(self, num_class, stride=None, mix_components=27, use_height=True, pred_facet=False):
        super(PicassoNetII,self).__init__()

        if stride is None:
            self.stride = [4, 3, 3, 2, 2]
        self.num_class = num_class
        self.num_clusters = mix_components
        self.useArea, self.wgtBnd = (True, 1.) # recommended hyper-parameters for mesh decimation
        self.temperature = 0.1
        self.bn_momentum = 0.99
        self.num_blocks = 5
        self.Niters = [2, 2, 4, 4, 4]
        self.radius = [0.2, 0.4, 0.8]
        self.EncodeChannels = [32, 64, 96, 128, 192, 256]
        self.useHeight = use_height
        if self.useHeight:
            geometry_in_channels = 12
        else:
            geometry_in_channels = 9
        self.predFacet = pred_facet

        self.min_nv = 100  # set the minimum vertex number for the coarsest layer
        self.stride = np.float32(self.stride)
        self.nv_limit = np.int32(np.cumprod(self.stride[::-1])[::-1]*self.min_nv)

        self.build_mesh_hierarchy = meshUtil.MeshHierarchy(self.stride, self.nv_limit,
                                                           self.useArea, self.wgtBnd)
        self.cluster = []
        for k in range(self.num_blocks+1):
            self.cluster.append(BuildF2VCoeff(self.num_clusters, tau=self.temperature,
                                              scope='cluster%d'%k))

        # FirstBlock
        self.Conv0 = FirstBlock([geometry_in_channels,3], self.EncodeChannels[0], self.num_clusters,
                                bn_momentum=self.bn_momentum, scope='conv0')

        # setting of convolution blocks in the encoder
        self.Block = []
        growth_rate = 32
        self.Block.append(EncoderMeshBlock(self.EncodeChannels[0], self.EncodeChannels[1], growth_rate,
                                           self.num_clusters, max_iter=self.Niters[0],
                                           bn_momentum=self.bn_momentum, scope='block1'))
        self.Block.append(EncoderMeshBlock(self.EncodeChannels[1], self.EncodeChannels[2], growth_rate,
                                           self.num_clusters, max_iter=self.Niters[1],
                                           bn_momentum=self.bn_momentum, scope='block2'))
        self.Block.append(DualBlock(self.EncodeChannels[2], self.EncodeChannels[3], growth_rate,
                                    self.num_clusters, radius=self.radius[0], max_iter=self.Niters[2],
                                    bn_momentum=self.bn_momentum, scope='block3'))
        self.Block.append(DualBlock(self.EncodeChannels[3], self.EncodeChannels[4], growth_rate,
                                    self.num_clusters, radius=self.radius[1], max_iter=self.Niters[3],
                                    bn_momentum=self.bn_momentum, scope='block4'))
        self.Block.append(DualBlock(self.EncodeChannels[4], self.EncodeChannels[5], growth_rate,
                                    self.num_clusters, radius=self.radius[2], max_iter=self.Niters[4],
                                    bn_momentum=self.bn_momentum, scope='block5'))

        # reducing channels of low-level features in the encoder
        self.Decode = []
        self.DecodeChannels = [self.EncodeChannels[-1], 128, 128, 96, 96, 96]
        for k in range(self.num_blocks):
            m = self.num_blocks - k - 1
            self.Decode.append(PerItemConv3d(self.EncodeChannels[m]+self.DecodeChannels[k],
                                             self.DecodeChannels[k+1], bn_momentum=self.bn_momentum,
                                             scope='decode%d'%m))
            print(self.EncodeChannels[m]+self.DecodeChannels[k], self.DecodeChannels[k+1])

        # mesh pooling, unpooling configuration
        self.Mesh_Pool = mesh_pool(method='max')
        self.Mesh_Unpool = mesh_unpool()

        # classifier
        if self.predFacet:
            self.V2F0 = V2FConv3d(96, 96, scope='V2F0_pred')
        self.Predict = PerItemConv3d(96, self.num_class, scope='prediction',
                                     with_bn=False, activation_fn=None)

    def extract_facet_geometric_features(self, vertex, face, normals):
        V1 = tf.gather(vertex, face[:, 0])
        V2 = tf.gather(vertex, face[:, 1])
        V3 = tf.gather(vertex, face[:, 2])

        Height_features = tf.stack([V1[:,2], V2[:,2], V3[:,2]], axis=1)

        Delta12 = V2[:,:3] - V1[:,:3]
        Delta23 = V3[:,:3] - V2[:,:3]
        Delta31 = V1[:,:3] - V3[:,:3]

        # length of edges
        L12 = tf.norm(Delta12, axis=-1, keepdims=True)
        L23 = tf.norm(Delta23, axis=-1, keepdims=True)
        L31 = tf.norm(Delta31, axis=-1, keepdims=True)
        Length_features = tf.concat([L12, L23, L31], axis=-1)

        # angles between edges
        Theta1 = tf.reduce_sum(Delta12*(-Delta31), axis=-1, keepdims=True)/(L12*L31)
        Theta2 = tf.reduce_sum((-Delta12)*Delta23, axis=-1, keepdims=True)/(L12*L23)
        Theta3 = tf.reduce_sum((-Delta23)*Delta31, axis=-1, keepdims=True)/(L23*L31)
        Theta_features = tf.concat([Theta1, Theta2, Theta3], axis=-1)

        if self.useHeight:
            facet_geometrics = tf.concat([Length_features, Theta_features,
                                          normals, Height_features], axis=-1)
        else:
            facet_geometrics = tf.concat([Length_features, Theta_features,
                                          normals], axis=-1)
        return facet_geometrics

    @staticmethod
    def get_face_normals(geometry_in, shuffle_normals=None):
        if shuffle_normals:
            sign = tf.random.uniform(tf.shape(geometry_in[:,:1]), 0, 1)
            sign = tf.where(sign<0.5, -1.0, 1.0)
            face_normals = sign*geometry_in[:,:3]
        else:
            face_normals = geometry_in[:,:3]
        return face_normals

    def __call__(self, vertex_in, face_in, nv_in, mf_in, facet_textures, bary_coeff, num_texture,
                 is_training=None, shuffle_normals=True):
        vertex_input = vertex_in
        vertex_in = vertex_in[:,:3]

        # build mesh hierarchy
        mesh_hierarchy = self.build_mesh_hierarchy(vertex_in, face_in, nv_in, mf_in)

        # import open3d as o3d
        # for i in range(6):
        #     vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[i][:4]
        #     mesh = o3d.geometry.TriangleMesh()
        #     mesh.vertices = o3d.utility.Vector3dVector(vertex_in[:,:3].numpy())
        #     mesh.triangles = o3d.utility.Vector3iVector(face_in.numpy())
        #     mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_in[:,3:].numpy())
        #     print(vertex_in.shape, face_in.shape)
        #     o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

        decoder_helper = []
        # ============================================Initial Conv==============================================
        vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[0][:4]
        full_vt_map = tf.range(tf.shape(vertex_in)[0], dtype=tf.int32)
        full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
        face_normals = self.get_face_normals(geometry_in, shuffle_normals=shuffle_normals)
        facet_geometrics = self.extract_facet_geometric_features(vertex_input, face_in, face_normals)
        filt_coeff = self.cluster[0](face_normals=face_normals)
        feats = self.Conv0(facet_geometrics, facet_textures, bary_coeff, num_texture, face_in,
                           full_nf_count, full_vt_map, filt_coeff, is_training=is_training)

        decoder_helper.append(feats)
        vertex_out, _, _, _, _, vt_replace, vt_map = mesh_hierarchy[1]
        pooled_feats = self.Mesh_Pool(feats, vt_replace, vt_map, vertex_out)  # mesh pool0
        # ===============================================The End================================================

        # ============================================Encoder Flow==============================================
        for k in range(self.num_blocks):
            # block computation:
            vertex_in, face_in, geometry_in, nv_in = mesh_hierarchy[k+1][:4]
            full_vt_map = tf.range(tf.shape(vertex_in)[0], dtype=tf.int32)
            full_nf_count = meshUtil.count_vertex_adjface(face_in, full_vt_map, vertex_in)
            face_normals = self.get_face_normals(geometry_in, shuffle_normals=shuffle_normals)
            filt_coeff = self.cluster[k+1](face_normals=face_normals)
            feats = self.Block[k](pooled_feats, vertex_in, face_in, full_nf_count, full_vt_map, filt_coeff,
                                  nv_in, is_training=is_training)

            if k<(self.num_blocks-1):
                decoder_helper.append(feats)
                vertex_out, _, _, _, _, vt_replace, vt_map = mesh_hierarchy[k+2]
                pooled_feats = self.Mesh_Pool(feats, vt_replace, vt_map, vertex_out)  # mesh pool
        # ===============================================The End================================================

        # ===============================combine with low-level encoder features================================
        for k in range(self.num_blocks):
            # upsampling and feature combination with Encoder features
            it = -k + self.num_blocks
            vt_replace, vt_map = mesh_hierarchy[it][-2:]
            low_level_feats = decoder_helper[it-1]
            upsampled_feats = self.Mesh_Unpool(feats, vt_replace, vt_map)
            feats = tf.concat([low_level_feats, upsampled_feats], axis=-1)
            feats = self.Decode[k](feats, is_training=is_training)
        # ===============================================The End================================================

        if self.predFacet:
            face_in = mesh_hierarchy[0][1]
            feats = self.V2F0(feats, face_in)
        pred_logits = self.Predict(feats)
        return pred_logits