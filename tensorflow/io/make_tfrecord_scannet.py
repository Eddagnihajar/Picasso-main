import numpy as np
import os, sys
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import open3d as o3d
from termcolor import colored
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--read_dir', required=True, help='path to mesh data')
parser.add_argument('--write_dir', required=True, help='path to tfrecord data')
opt = parser.parse_args()

# labels of ignored an interested classes
ignored_cls = 0
interest_cls = [*range(1,13), 14, 16, 24, 28, 33, 34, 36, 39]
num_classes = len(interest_cls)
new_labels = [*range(num_classes)]
print('ignored_cls:',ignored_cls)
print('interest_cls:',interest_cls)
print('new_labels:',new_labels)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_semseg_labels(scenePath, scene_name):
    semantic_file = os.path.join(scenePath, '%s_semantic_labels.txt'%scene_name)
    nyu40_labels = np.loadtxt(semantic_file, dtype=np.int32)
    uni_ids, cnts = np.unique(nyu40_labels, return_counts=True)
    print(uni_ids)

    nyu40_labels[nyu40_labels>40] = ignored_cls

    # check labels
    if np.any(uni_ids<0):
        raise ValueError('Minimum label should be zero but negative values are detected!')

    # process labels
    scan20_labels = -np.ones_like(nyu40_labels)  # use -1 for un-assigned labels
    for id in uni_ids:
        index = (nyu40_labels==id)
        if id==ignored_cls:
            continue
        elif id in interest_cls:
            scan20_labels[index] = interest_cls.index(id)
        else:
            scan20_labels[index] = 20

    assert (np.all((scan20_labels>=-1) & (scan20_labels<=20)))
    return nyu40_labels, scan20_labels


def make_tfrecord_seg(scenePath, phase, store_folder="", visualize=False):
    scene_name = os.path.basename(scenePath)

    print("========================make tfrecords of scannet %s======================="%scene_name)
    print("start to make %s.tfrecord:"%scene_name)
    # ===========================Load Meshes===========================
    # Note: open3d will normalize the mesh colors to range [0,1] automatically
    # RGB regenerated from the extracted RGBD frames
    meshPath = os.path.join(scenePath, '%s_vh_clean_2.ply'%scene_name)
    mesh = o3d.io.read_triangle_mesh(meshPath)
    if visualize:
        o3d.visualization.draw_geometries([mesh],mesh_show_back_face=True)
    xyz = np.asarray(mesh.vertices,dtype=np.float32)
    rgb = np.asarray(mesh.vertex_colors,dtype=np.float32)
    face = np.asarray(mesh.triangles,dtype=np.int32)
    # =================================================================

    # ================get semantic segmentation labels=================
    if phase!='test':  # get the labels
        nyu40_labels, seg_labels = get_semseg_labels(scenePath,scene_name)
        assert (xyz.shape[0]==nyu40_labels.shape[0])
    else:
        seg_labels = -np.ones(shape=[xyz.shape[0]], dtype=np.int32)
        nyu40_labels = seg_labels
    # =================================================================
    assert(np.amin(rgb)>=0)
    assert(np.amax(rgb)<=1)

    # =====================location normalization======================
    xyz_min = np.amin(xyz, axis=0, keepdims=True)
    xyz_max = np.amax(xyz, axis=0, keepdims=True)
    xyz_center = (xyz_min+xyz_max)/2
    xyz_center[0][-1] = xyz_min[0][-1]
    xyz -= xyz_center  # align to room bottom center
    # =================================================================
    print('%s, %d vertices, %d faces'%(scene_name,xyz.shape[0],face.shape[0]))

    filename = os.path.join(store_folder, phase, '%s.tfrecord'%(scene_name))
    if not os.path.exists(filename):
        writer = tf.io.TFRecordWriter(filename)

        xyz_raw = xyz.tobytes()
        rgb_raw = rgb.tobytes()
        face_raw = face.tobytes() # no augmentation is applied to face
        seg_label_raw = seg_labels.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
                                    'xyz_raw': _bytes_feature(xyz_raw),
                                    'rgb_raw': _bytes_feature(rgb_raw),
                                    'face_raw': _bytes_feature(face_raw),
                                    'seg_label_raw': _bytes_feature(seg_label_raw)}))

        writer.write(example.SerializeToString())
        writer.close()
    print("======================================The End=======================================\n")

    return


if __name__=='__main__':
    store_folder = opt.write_dir
    if not os.path.exists(store_folder):
        os.mkdir(store_folder)

    Lists = {}
    Lists['train'] = [line.rstrip() for line in open(os.path.join(opt.read_dir, 'scannetv2_train.txt'))]
    Lists['val']   = [line.rstrip() for line in open(os.path.join(opt.read_dir, 'scannetv2_val.txt'))]
    Lists['test']  = [line.rstrip() for line in open(os.path.join(opt.read_dir, 'scannetv2_test.txt'))]

    for phase in ['train','val','test']:
        fileList = Lists[phase]
        if phase=='test':
            folder = 'scans_test'
        else:
            folder = 'scans_trainval'

        if not os.path.exists(os.path.join(store_folder,phase)):
            os.makedirs(os.path.join(store_folder,phase))

        for scene_name in fileList:
            scenePath = os.path.join(opt.read_dir,folder,scene_name)
            make_tfrecord_seg(scenePath, phase, store_folder=store_folder, visualize=False)

    TFRecord = {}
    TFRecord['train'] = glob.glob(os.path.join(store_folder, 'train', '*.tfrecord'))
    TFRecord['val'] = glob.glob(os.path.join(store_folder, 'val', '*.tfrecord'))
    TFRecord['test'] = glob.glob(os.path.join(store_folder, 'test', '*.tfrecord'))
    trainfile = open(os.path.join(store_folder, 'train_files.txt'), 'w')
    valfile = open(os.path.join(store_folder, 'val_files.txt'), 'w')
    testfile = open(os.path.join(store_folder, 'test_files.txt'), 'w')
    for filepath in TFRecord['train']:
        trainfile.write("%s\n" % filepath)
    for filepath in TFRecord['val']:
        valfile.write("%s\n" % filepath)
    for filepath in TFRecord['test']:
        testfile.write("%s\n" % filepath)
    trainfile.close()
    valfile.close()
    testfile.close()


