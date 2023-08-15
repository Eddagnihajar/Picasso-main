import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import picasso.mesh.utils as meshUtil
from picasso.augmentor import Augment
from picasso.mesh.dataset import Dataset as MeshDataset
from picasso.models.scene_seg_texture import PicassoNetII

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_scannet_plain', help='Log dir [default: log_scannet_plain]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--max_num_vertices', type=int, default=1500000, help='maximum vertices allowed in a batch')
parser.add_argument('--voxel_size', type=int, default=2, help='Voxel Size to downsample input mesh')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
parser.add_argument('--num_augment', type=int, default=20, help='number of augmentations')
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

# import importlib.util
# spec = importlib.util.spec_from_file_location("PicassoNetII", LOG_DIR + "/scene_seg_texture.py")
# foo = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(foo)
# PicassoNetII = foo.PicassoNetII

# labels, classnames, and colormaps
NUM_CLASSES = 20
labels = [*range(NUM_CLASSES)]
classnames = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
              'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refridgerator',
              'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
colormap = [(174, 199, 232), (152, 223, 138), (31, 119, 180),  (255, 187, 120),
            (188, 189, 34),  (140, 86, 75),   (255, 152, 150), (214, 39, 40),
            (197, 176, 213), (148, 103, 189), (196, 156, 148), (23, 190, 207),
            (247, 182, 210), (219, 219, 141), (255, 127, 14), (158, 218, 229),
            (44, 160, 44),   (112, 128, 144), (227, 119, 194), (82, 84, 163)]

# tfrecord file_lists
Lists = {}
Lists['val']   =  [line.rstrip() for line in open(os.path.join(opt.data_dir, 'val_files.txt'))]
Lists['test']  =  [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files.txt'))]


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
    # augment_rgb = Augment.random_drop_color(augment_rgb, prob=prob)
    augment_rgb = Augment.shift_color(augment_rgb, prob=prob)
    augment_rgb = Augment.jitter_color(augment_rgb, prob=prob)
    augment_rgb = Augment.auto_contrast_color(augment_rgb, prob=0.5)

    vertex = augment_xyz
    texture = augment_rgb
    return vertex, texture


def parse_fn(item, is_training=True):
    features = {
        'xyz_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'rgb_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'face_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'seg_label_raw': tf.io.FixedLenFeature([], dtype=tf.string)
    }
    features = tf.io.parse_single_example(item, features=features)

    dense_xyz = tf.io.decode_raw(features['xyz_raw'], tf.float32)
    dense_rgb = tf.io.decode_raw(features['rgb_raw'], tf.float32)
    dense_face = tf.io.decode_raw(features['face_raw'], tf.int32)
    dense_semantic_label = tf.io.decode_raw(features['seg_label_raw'], tf.int32)

    dense_xyz = tf.reshape(dense_xyz, [-1,3])
    dense_rgb = tf.reshape(dense_rgb, [-1,3])
    dense_face = tf.reshape(dense_face, [-1,3])
    dense_label = tf.reshape(dense_semantic_label, [-1])

    dense_vertex = tf.concat([dense_xyz, dense_rgb], axis=-1)
    assert(dense_vertex.shape[0]==dense_label.shape[0])

    dense_nv, dense_mf = tf.shape(dense_vertex[:, 0]), tf.shape(dense_face[:, 0])
    return dense_vertex, dense_face, dense_nv, dense_mf, dense_label
    # =======================================================================================================


class MyModel(tf.Module):
    def __init__(self, net):
        '''
            Setting all the variables for our model.
        '''
        super(MyModel, self).__init__()
        self.model = net

    def evaluate(self, test):       
        class_names = classnames[:20]
        total_seen_class = {cat: 0 for cat in class_names}
        total_correct_class = {cat: 0 for cat in class_names}
        total_union_class = {cat: 0 for cat in class_names}

        iter, batch_idx, test_time = 0, 0, 0.0
        for dense_room_meshes, dense_room_labels in test.batch(1):
            dense_vertex, dense_face, dense_nv, dense_mf = dense_room_meshes
            dense_label = dense_room_labels

            dense_room_logits = tf.zeros(shape=[dense_vertex.shape[0], NUM_CLASSES])
            for augIter in range(opt.num_augment):
                if augIter>0: # augment the mesh
                    dense_xyzAug, dense_rgbAug = augment_fn(dense_vertex[:,:3], dense_vertex[:,3:])
                    dense_vertexAug = tf.concat([dense_xyzAug, dense_rgbAug], axis=-1)
                else:
                    dense_vertexAug = dense_vertex

                # ===================================voxelization===================================
                # voxelize the dense mesh with user-defined voxel_size size, here we use 2cm
                vertex, face, label, \
                voxel2dense_idx = meshUtil.voxelize_mesh(dense_vertexAug, dense_face,
                                                         voxel_size=opt.voxel_size,
                                                         seg_labels=dense_label,
                                                         return_inverse=True)
                face_texture = tf.concat([tf.gather(vertex[:,3:], face[:,0]),
                                          tf.gather(vertex[:,3:], face[:,1]),
                                          tf.gather(vertex[:,3:], face[:,2])], axis=-1)
                face_texture = tf.reshape(face_texture, [-1, 3])
                bary_coeff = tf.tile(tf.eye(3), [face.shape[0], 1])
                num_texture = 3*tf.ones(shape=[face.shape[0]], dtype=tf.int32)
                assert (vertex.shape[0]==label.shape[0])
                nv, mf = tf.shape(vertex[:,0]), tf.shape(face[:,0])
                # ==================================================================================

                if augIter>0: # augment the mesh
                    room_logits  = self.model(vertex, face, nv, mf, face_texture, bary_coeff,
                                              num_texture, is_training=False, shuffle_normals=True)
                else:
                    room_logits  = self.model(vertex, face, nv, mf, face_texture, bary_coeff,
                                              num_texture, is_training=False, shuffle_normals=False)
                    room_logits += self.model(vertex, face, nv, mf, face_texture, bary_coeff,
                                              num_texture, is_training=False, shuffle_normals=True)
                dense_room_logits += tf.gather(room_logits, voxel2dense_idx)

            iter = iter + 1
            fname = os.path.basename(Lists['val'][iter-1]).split('.')[0]
            print('%d/%d processed, %s' % (iter, len(Lists['val']), fname))

            gt_labels = dense_label.numpy()
            pred_labels = tf.math.argmax(dense_room_logits, axis=-1).numpy()

            nyu40_gt_labels = np.ones_like(gt_labels)*40
            nyu40_pred_labels = np.ones_like(gt_labels)*40
            VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
            for i in range(NUM_CLASSES):
                nyu40_gt_labels[gt_labels==i] = VALID_CLASS_IDS[i]
                nyu40_pred_labels[pred_labels==i] = VALID_CLASS_IDS[i]
            np.savetxt('%s/Pred/%s.txt' % (results_folder, fname), nyu40_pred_labels, fmt='%d', delimiter=',')
            np.savetxt('%s/GT/%s.txt' % (results_folder, fname), nyu40_gt_labels, fmt='%d', delimiter=',')

            interest_index = tf.where((dense_label>=0) & (dense_label<20))
            gt_labels = tf.gather_nd(dense_label, interest_index)
            pred_labels = tf.math.argmax(tf.gather_nd(dense_room_logits, interest_index), axis=-1)
            gt_labels = gt_labels.numpy()
            pred_labels = pred_labels.numpy()
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
        log_string('eval mIoU of all 20 classes:\t %3.2f'%(100*np.mean(list(class_iou.values()))))

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
    valSet = MeshDataset(Lists['val']).map(parse_fn, is_training=tf.constant(False))
    testSet = MeshDataset(Lists['test']).map(parse_fn, is_training=tf.constant(False))

    # create model & Make a loss object
    picasso_net = PicassoNetII(num_class=NUM_CLASSES, mix_components=opt.num_clusters, use_height=True)

    # Create an instance of the model
    model = MyModel(net=picasso_net)

    # create a checkpoint that will manage all the objects with trackable state
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, trainable=False), net=picasso_net)

    print("LOG_DIR=", LOG_DIR)
    manager = tf.train.CheckpointManager(ckpt, LOG_DIR + '/tf_ckpts', max_to_keep=300)
    model.fit(test=valSet, manager=manager)

