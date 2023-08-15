import numpy as np
import tensorflow as tf
import picasso.point.utils as pointUtil
from picasso.point.layers import PCloudConv3d
from picasso.mesh.layers import V2VConv3d, F2VConv3d, F2FConv3d, PerItemConv3d


class DualBlock(tf.Module):
    def __init__(self, in_channels, out_channels, growth_rate, mesh_kernel_size, radius,
                 max_iter=2, max_nn=None, scope='', bn_momentum=None):
        super(DualBlock, self).__init__()
        self.radius = radius
        self.max_nn = max_nn
        self.maxIter = max_iter

        self.pcloud_kernel = [8, 4, 1]  # keep as default

        factor = 4
        self.DualConvGroup = []
        _in_channels_ = in_channels
        for n in range(0,self.maxIter):
            MeshConv1 = V2VConv3d(_in_channels_, [factor*growth_rate,factor*growth_rate],
                                  mesh_kernel_size, dual_depth_multiplier=[1,1],
                                  scope=scope+'/meshConv%d_1'%(n+1),
                                  bn_momentum=bn_momentum)
            MeshConv2 = V2VConv3d(factor*growth_rate, [factor*growth_rate,2*growth_rate],
                                  mesh_kernel_size, dual_depth_multiplier=[1,1],
                                  scope=scope+'/meshConv%d_2'%(n+1),
                                  bn_momentum=bn_momentum)
            Dense = PerItemConv3d(_in_channels_, factor*growth_rate,
                                  scope=scope+'/pointwise%d'%(n+1),
                                  bn_momentum=bn_momentum)
            PointConv1 = PCloudConv3d(factor*growth_rate, 2*growth_rate,
                                      np.prod(self.pcloud_kernel)+1,
                                      scope=scope+'/pointConv%d_1'%(n+1),
                                      bn_momentum=bn_momentum)
            self.DualConvGroup.append(MeshConv1)
            self.DualConvGroup.append(MeshConv2)
            self.DualConvGroup.append(Dense)
            self.DualConvGroup.append(PointConv1)
            _in_channels_ += (2*growth_rate*2)

        self.Transit = PerItemConv3d(_in_channels_, out_channels, scope=scope+'/pointwise',
                                     bn_momentum=bn_momentum)

    def __call__(self, inputs, vertex, face, full_nf_count, full_vt_map, filt_coeff,
                 nv_in, is_training=None):
        # build graph for point-based convolution
        xyz = vertex[:,:3]
        graph_tuple = self._build_point_graph_(xyz, nv_in, nn_uplimit=self.max_nn)

        # perform the rest dualConv
        for n in range(0,self.maxIter):
            Mnet = self.DualConvGroup[4*n](inputs, face, full_nf_count, full_vt_map,
                                           filt_coeff, is_training)   # V2V
            Mnet = self.DualConvGroup[4*n+1](Mnet, face, full_nf_count, full_vt_map,
                                             filt_coeff, is_training) # V2V
            Pnet = self.DualConvGroup[4*n+2](inputs, is_training)
            Pnet = self.DualConvGroup[4*n+3](Pnet, *graph_tuple, is_training)  # Point
            inputs = tf.concat([inputs,Mnet,Pnet],axis=-1)

        outputs = self.Transit(inputs, is_training)
        return outputs

    def _build_point_graph_(self, xyz, nv_in, nn_uplimit=None):
        nn_cnt, nn_idx, nn_dst, \
        filt_idx, filt_coeff = pointUtil.build_range_graph(xyz_data=xyz, xyz_query=xyz,
                                                           nv_data=nv_in, nv_query=nv_in,
                                                           radius=self.radius, nn_uplimit=nn_uplimit,
                                                           kernel_shape=self.pcloud_kernel, fuzzy=True)
        return nn_cnt, nn_idx, filt_idx, filt_coeff


class EncoderMeshBlock(tf.Module):
    def __init__(self, in_channels, out_channels, growth_rate, mesh_kernel_size,
                 max_iter=2, scope='', bn_momentum=0.9):
        super(EncoderMeshBlock, self).__init__()
        self.maxIter = max_iter

        factor = 4
        self.MeshConvGroup = []
        _in_channels_ = in_channels
        for n in range(0,self.maxIter):
            MeshConv1 = V2VConv3d(_in_channels_, [factor*growth_rate,factor*growth_rate],
                                  mesh_kernel_size, dual_depth_multiplier=[1,1],
                                  scope=scope+'/meshConv%d_1'%(n+1),
                                  bn_momentum=bn_momentum)
            MeshConv2 = V2VConv3d(factor*growth_rate, [factor*growth_rate,growth_rate],
                                  mesh_kernel_size, dual_depth_multiplier=[1,1],
                                  scope=scope+'/meshConv%d_2'%(n+1),
                                  bn_momentum=bn_momentum)
            self.MeshConvGroup.append(MeshConv1)
            self.MeshConvGroup.append(MeshConv2)
            _in_channels_ += growth_rate

        self.Transit = PerItemConv3d(_in_channels_, out_channels, scope=scope+'/pointwise',
                                     bn_momentum=bn_momentum)

    def __call__(self, inputs, vertex, face, full_nf_count, full_vt_map, filt_coeff,
                 nv_in, is_training=None):
        for n in range(0, self.maxIter):
            net = self.MeshConvGroup[2*n](inputs, face, full_nf_count, full_vt_map, filt_coeff,
                                          is_training)
            net = self.MeshConvGroup[2*n+1](net, face, full_nf_count, full_vt_map, filt_coeff,
                                            is_training)
            inputs = tf.concat([inputs, net], axis=-1)

        outputs = self.Transit(inputs, is_training)
        return outputs


class FirstBlock(tf.Module):
    def __init__(self, in_channels, out_channels, mesh_kernel_size,
                 scope='', with_bn=True, bn_momentum=0.9):
        super(FirstBlock, self).__init__()
        self.MLP0 = PerItemConv3d(in_channels, out_channels, with_bn=with_bn,
                                  scope='mlp0', bn_momentum=bn_momentum)
        self.F2V0 = F2VConv3d(out_channels, out_channels, mesh_kernel_size,
                              depth_multiplier=2, scope=scope+'/F2V0',
                              with_bn=with_bn, bn_momentum=bn_momentum)

    def __call__(self, facet_geometrics, face, full_nf_count, full_vt_map, filt_coeff, is_training=None):
        net = self.MLP0(facet_geometrics, is_training=is_training)
        net = self.F2V0(net, face, full_nf_count, full_vt_map, filt_coeff, is_training=is_training)
        return net


class FirstBlockTexture(tf.Module):
    def __init__(self, in_channels, out_channels, mesh_kernel_size, scope='',
                 with_bn=True, bn_momentum=0.9):
        super(FirstBlockTexture, self).__init__()
        GeometryInChannels, TextureInChannels = in_channels
        self.F2F0 = F2FConv3d(TextureInChannels, out_channels, with_bn=with_bn,
                              scope='f2f0', bn_momentum=bn_momentum)
        self.MLP0 = PerItemConv3d(GeometryInChannels, out_channels, with_bn=with_bn,
                                  scope='mlp0', bn_momentum=bn_momentum)
        self.F2V0 = F2VConv3d(out_channels*2, out_channels, mesh_kernel_size,
                              depth_multiplier=1, scope=scope+'/F2V0',
                              with_bn=with_bn, bn_momentum=bn_momentum)

    def __call__(self, facet_geometrics, facet_textures, bary_coeff, num_texture,
                 face, full_nf_count, full_vt_map, filt_coeff, is_training=None):
        Tnet = self.F2F0(facet_textures, bary_coeff, num_texture, is_training=is_training)
        Gnet = self.MLP0(facet_geometrics, is_training=is_training)
        # net = Tnet + Gnet
        net = tf.concat([Tnet, Gnet],axis=-1)
        net = self.F2V0(net, face, full_nf_count, full_vt_map, filt_coeff, is_training=is_training)
        return net