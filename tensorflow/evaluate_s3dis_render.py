import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import open3d as o3d
from picasso.augmentor import Augment
from picasso.mesh.dataset import Dataset as MeshDataset


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_s3dis_render', help='Log dir [default: log_s3dis_render]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
parser.add_argument('--num_augment', type=int, default=20, help='number of test augmentations')
opt = parser.parse_args()

# set gpu
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu], True)

LOG_DIR = opt.log_dir
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

results_folder = LOG_DIR + '/test_results'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
    os.makedirs(results_folder+'/GT')
    os.makedirs(results_folder+'/Pred')
    os.makedirs(results_folder+'/Vertex')
    os.makedirs(results_folder+'/Logit')

import importlib.util
spec = importlib.util.spec_from_file_location("PicassoNetII", LOG_DIR + "/pi2_seg_texture.py")
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
PicassoNetII = foo.PicassoNetII

# tfrecord file_lists
Lists = {}
Lists['test']   =  [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files_fold5.txt'))]

NUM_CLASSES = 13
classnames = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
              'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def augment_fn(vertex, texture):
    prob = 1.0
    augment_xyz = vertex 
    augment_rgb = texture

    # augment (xyz) on the fly with spatial transformations
    augment_xyz = Augment.rotate_point_cloud(augment_xyz, upaxis=3, prob=prob)
    augment_xyz = Augment.rotate_perturbation_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.flip_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.random_scale_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.shift_point_cloud(augment_xyz, prob=prob)

    # augment (rgb) color on the fly
    augment_rgb = Augment.random_drop_color(augment_rgb, prob=prob)
    augment_rgb = Augment.shift_color(augment_rgb, prob=prob)
    augment_rgb = Augment.jitter_color(augment_rgb, prob=prob)
    augment_rgb = Augment.auto_contrast_color(augment_rgb, prob=0.5)

    vertex = augment_xyz
    texture = augment_rgb
    return vertex, texture


def parse_fn(item, is_training=True):
    features = {
        'xyz_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'face_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'normal_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'face_texture_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'bary_coeff_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'num_texture_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'seg_label_raw': tf.io.FixedLenFeature([], dtype=tf.string)
    }
    features = tf.io.parse_single_example(item, features=features)

    xyz = tf.io.decode_raw(features['xyz_raw'], tf.float32)
    face = tf.io.decode_raw(features['face_raw'], tf.int32)
    face_texture = tf.io.decode_raw(features['face_texture_raw'], tf.float32)
    bary_coeff = tf.io.decode_raw(features['bary_coeff_raw'], tf.float32)
    num_texture = tf.io.decode_raw(features['num_texture_raw'], tf.int32)
    label = tf.io.decode_raw(features['seg_label_raw'], tf.int32)

    vertex = tf.reshape(xyz, [-1,3])
    face = tf.reshape(face, [-1,3])
    face_texture = tf.reshape(face_texture, [-1,3])
    bary_coeff = tf.reshape(bary_coeff, [-1,3])
    num_texture = tf.reshape(num_texture, [-1])
    label = tf.reshape(label, [-1])

    num_labelled = tf.reduce_sum(tf.cast((label>=0) & (label<NUM_CLASSES),dtype=tf.int32))
    nv, mf = tf.shape(vertex[:,0]), tf.shape(face[:,0])
    ratio = num_labelled/nv[0]
    if is_training and (ratio<0.2):
        return None
    else:
        return vertex, face, nv, mf, face_texture, bary_coeff, num_texture, label
    # =======================================================================================================


class MyModel(tf.Module):
    def __init__(self, net):
        '''
            Setting all the variables for our model.
        '''
        super(MyModel, self).__init__()
        self.seg_model = net

    def evaluate(self, testSet):
        class_names = classnames[:NUM_CLASSES]
        total_seen_class = {cat: 0 for cat in class_names}
        total_correct_class = {cat: 0 for cat in class_names}
        total_union_class = {cat: 0 for cat in class_names}

        iter, batch_idx, test_time = 0, 0, 0.0
        for room_meshes, room_labels in testSet.batch_render(1):
            vertex, face, nv, mf, \
            face_texture, baryCoeff, num_texture = room_meshes
            label = room_labels

            room_logits = tf.zeros(shape=[vertex.shape[0], NUM_CLASSES])
            for augIter in range(opt.num_augment):
                if augIter>0:  # augment the mesh
                    vertexAug, textureAug = augment_fn(vertex, face_texture)
                else:
                    vertexAug, textureAug = vertex, face_texture

                if augIter>0:  # augment the mesh
                    room_logits += self.seg_model(vertexAug, face, nv, mf, textureAug, baryCoeff, num_texture,
                                                 is_training=tf.constant(False), shuffle_normals=True)
                else:
                    room_logits = self.seg_model(vertexAug, face, nv, mf, textureAug, baryCoeff, num_texture,
                                                 is_training=tf.constant(False), shuffle_normals=False)
                    room_logits += self.seg_model(vertexAug, face, nv, mf, textureAug, baryCoeff, num_texture,
                                                  is_training=tf.constant(False), shuffle_normals=True)

            interest_index = tf.where((label>=0) & (label<NUM_CLASSES))
            gt_labels = tf.gather_nd(label, interest_index)
            pred_labels = tf.math.argmax(tf.gather_nd(room_logits, interest_index), axis=-1)

            iter = iter + 1
            fname = os.path.basename(Lists['test'][iter-1]).split('.')[0]
            print('%d/%d processed, %s'%(iter, len(Lists['test']), fname))

            gt_labels = gt_labels.numpy()
            pred_labels = pred_labels.numpy()
            vertex = vertex.numpy()

            np.savetxt("%s/Vertex/%s.txt"% (results_folder, fname), vertex, fmt='%.3f', delimiter=',')
            np.savetxt('%s/Pred/%s.txt' % (results_folder, fname), pred_labels, fmt='%d', delimiter=',')
            np.savetxt('%s/GT/%s.txt' % (results_folder, fname), gt_labels, fmt='%d', delimiter=',')
            np.savetxt('%s/Logit/%s.txt' % (results_folder, fname), room_logits, fmt='%.3f', delimiter=',')

            for l, cat in enumerate(class_names):
                total_seen_class[cat] += np.sum(gt_labels==l)
                total_union_class[cat] += (np.sum((pred_labels==l) | (gt_labels==l)))
                total_correct_class[cat] += (np.sum((pred_labels==l) & (gt_labels==l)))

        class_iou = {cat: 0.0 for cat in class_names}
        class_acc = {cat: 0.0 for cat in class_names}
        for cat in class_names:
            class_iou[cat] = total_correct_class[cat]/(float(total_union_class[cat])+np.finfo(float).eps)
            class_acc[cat] = total_correct_class[cat]/(float(total_seen_class[cat])+np.finfo(float).eps)

        total_correct = sum(list(total_correct_class.values()))
        total_seen = sum(list(total_seen_class.values()))
        log_string('total number of test sample:\t %d'%iter)
        log_string('eval overall class accuracy:\t %d/%d=%3.2f'%(total_correct, total_seen,
                                                          100*total_correct/float(total_seen)))
        log_string('eval average class accuracy:\t %3.2f'%(100*np.mean(list(class_acc.values()))))
        for cat in class_names:
            log_string('eval mIoU of %14s:\t %3.2f'%(cat, 100*class_iou[cat]))
        log_string('eval mIoU of all 13 classes:\t %3.2f'%(100*np.mean(list(class_iou.values()))))

    def fit(self, test, manager):
        '''
            This fit function runs training and testing.
        '''
        checkpoint = os.path.dirname(manager.latest_checkpoint) + '/ckpt-%d'%opt.ckpt_epoch
        ckpt.restore(checkpoint)
        print("Restored from {}".format(checkpoint))

        start_time = time.time()
        self.evaluate(test)
        end_time = time.time() - start_time
        print('time taken to evaluate is %.3f seconds'%end_time)


if __name__=='__main__':
    testSet = MeshDataset(Lists['test']).map(parse_fn, is_training=False)

    # create model
    picasso_net = PicassoNetII(num_class=NUM_CLASSES, mix_components=opt.num_clusters, use_height=True)

    # Create an instance of the model
    model = MyModel(net=picasso_net)

    # create a checkpoint that will manage all the objects with trackable state
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, trainable=False), net=picasso_net)

    print("LOG_DIR=", LOG_DIR)
    manager = tf.train.CheckpointManager(ckpt, LOG_DIR+'/tf_ckpts', max_to_keep=200)
    model.fit(test=testSet, manager=manager)

