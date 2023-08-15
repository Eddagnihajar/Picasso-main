import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import open3d as o3d
from picasso.augmentor import Augment
from picasso.mesh.dataset import Dataset as MeshDataset
from picasso.models.pi2_cls import PicassoNetII

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_shapenetcore', help='Log dir [default: ../log_shapenetcore]')
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
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp train_ShapeNetCore.py %s'%(LOG_DIR))   # bkp of train procedure
os.system('cp picasso/models/pi2_cls.py %s'%(LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

NUM_CLASSES = 55

# tfrecord file_lists
Lists = {}
Lists['test']  = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files.txt'))]
print("numbers of trainval and test samples are %d and %d, respectively."%(len(Lists['train']),
                                                                           len(Lists['test'])))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def augment_fn(xyz):
    prob = 1.0
    augment_xyz = xyz
    augment_xyz = Augment.flip_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.random_scale_point_cloud(augment_xyz, scale_low=0.5, scale_high=1.5, prob=prob)
    augment_xyz = Augment.rotate_point_cloud(augment_xyz, upaxis=3, prob=prob)
    augment_xyz = Augment.rotate_perturbation_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.jitter_point_cloud(augment_xyz, sigma=0.01, prob=prob)
    return augment_xyz


def parse_fn(item, is_training=True):
    features = {
        'xyz_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'face_raw': tf.io.FixedLenFeature([], dtype=tf.string),
        'label_raw': tf.io.FixedLenFeature([], dtype=tf.int64)
    }
    features = tf.io.parse_single_example(item, features=features)

    xyz   = tf.io.decode_raw(features['xyz_raw'], tf.float32)
    face  = tf.io.decode_raw(features['face_raw'], tf.int32)
    label = tf.cast(features['label_raw'], tf.int32)

    assert(label>=0 and label<NUM_CLASSES)
    xyz = tf.reshape(xyz, [-1, 3])
    face = tf.reshape(face, [-1, 3])

    xyz_min = tf.reduce_min(xyz, axis=0, keepdims=True)
    xyz_max = tf.reduce_max(xyz, axis=0, keepdims=True)
    scale = tf.reduce_max(xyz_max - xyz_min)/2
    xyz /= scale

    nv, mf = tf.shape(xyz[:,0]), tf.shape(face[:,0])
    return xyz, face, nv, mf, label
    # =======================================================================================================


class MyModel(tf.Module):
    def __init__(self, net):
        '''
            Setting all the variables for our model.
        '''
        super(MyModel, self).__init__()
        self.model = net

    def evaluate(self, testSet):
        test = MeshDataset(Lists['test']).map(parse_fn, tf.constant(False))

        gt_labels, pred_labels = [], []
        for meshes, labels in testSet.batch(1):
            vertex, face, nv, mf = meshes

            pred_logits = tf.zeros([1,NUM_CLASSES])
            for augIter in range(opt.num_augment):
                if augIter>0:  # augment the mesh
                    vertexAug = augment_fn(vertex)
                else:
                    vertexAug = vertex

                if augIter>0:  # augment the mesh
                    pred_logits += self.model(vertexAug, face, nv, mf, is_training=False,
                                              shuffle_normals=True)
                else:
                    pred_logits = self.model(vertexAug, face, nv, mf, is_training=False,
                                             shuffle_normals=False)
                    pred_logits += self.model(vertexAug, face, nv, mf, is_training=False,
                                              shuffle_normals=True)
            predLabel = np.argmax(pred_logits)
            pred_labels.append(predLabel)
            gt_labels.append(labels)

        gt_labels = np.asarray(gt_labels)
        pred_labels = np.asarray(pred_labels)
        num_correct = np.sum(gt_labels==pred_labels)
        print('classification accuracy is %d/%d=%3.2f%%'%(num_correct, gt_labels.shape[0],
                                                          num_correct/gt_labels.shape[0]*100))

    def fit(self, test, manager):
        '''
            This fit function runs training and testing.
        '''
        checkpoint = os.path.dirname(manager.latest_checkpoint) + '/ckpt-%d' % opt.ckpt_epoch
        ckpt.restore(checkpoint)
        print("Restored from {}".format(checkpoint))

        start_time = time.time()
        self.evaluate(test)
        end_time = time.time() - start_time
        print('time taken to evaluate is %.3f seconds' % end_time)


if __name__=='__main__':
    # load in datasets
    testSet = MeshDataset(Lists['test']).map(parse_fn, is_training=tf.constant(False))

    # create model & Make a loss object
    picasso_net = PicassoNetII(num_class=NUM_CLASSES, stride=[3,2,2,2], mix_components=opt.num_clusters)

    # Create an instance of the model
    model = MyModel(net=picasso_net)

    # create a checkpoint that will manage all the objects with trackable state
    ckpt = tf.train.Checkpoint(step=tf.Variable(1, trainable=False), net=picasso_net)
    manager = tf.train.CheckpointManager(ckpt, LOG_DIR+'/tf_ckpts', max_to_keep=300)
    model.fit(test=testSet, manager=manager)


