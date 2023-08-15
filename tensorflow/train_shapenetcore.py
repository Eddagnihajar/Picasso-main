import warnings
warnings.filterwarnings("ignore")
import argparse, time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import open3d as o3d
from datetime import datetime
import picasso.mesh.utils as meshUtil
from picasso.augmentor import Augment
from picasso.mesh.dataset import Dataset as MeshDataset
from picasso.models.shape_cls import PicassoNetII

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_shapenetcore', help='Log dir [default: ../log_shapenetcore]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--max_num_vertices', type=int, default=1600000, help='maximum vertices allowed in a batch')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
opt = parser.parse_args()

# set gpu
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu], True)

LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp train_ShapeNetCore.py %s'%(LOG_DIR))   # bkp of train procedure
os.system('cp picasso/models/shape_cls.py %s'%(LOG_DIR))
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

NUM_CLASSES = 55

# tfrecord file_lists
Lists = {}
Lists['train'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'train_files.txt'))]
Lists['val']   = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'val_files.txt'))]
Lists['test']  = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files.txt'))]
Lists['train'] =  Lists['train'] + Lists['val']
print("numbers of trainval and test samples are %d and %d, respectively."%(len(Lists['train']),len(Lists['test'])))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def augment_fn(xyz):
    prob = 0.5
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

    if is_training:
        xyz = augment_fn(xyz)

        # random drop out vertices with probability 0.5
        xyz, face, _, face_mask = Augment.random_drop_vertex(xyz, face, drop_rate=0.15, prob=0.5)

    nv, mf = tf.shape(xyz[:,0]), tf.shape(face[:,0])
    return xyz, face, nv, mf, label
    # =======================================================================================================


class MyModel(tf.Module):
    def __init__(self, net, optimizer, train_loss, train_metric,
                 test_loss, test_metric, train_writer, test_writer):
        '''
            Setting all the variables for our model.
        '''
        super(MyModel, self).__init__()
        self.model = net
        self.optimizer = optimizer
        self.train_loss = train_loss
        self.train_metric = train_metric
        self.test_loss = test_loss
        self.test_metric = test_metric
        self.train_writer = train_writer
        self.test_writer = test_writer

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, meshes, labels):
        vertex_in, face_in, nv_in, mf_in = meshes
        vertex_in = vertex_in[:tf.reduce_sum(nv_in),:] # change input axis-0 size to None
        face_in = face_in[:tf.reduce_sum(mf_in),:]     # change input axis-0 size to None
        with tf.GradientTape() as tape:
            pred_logits = self.model(vertex_in, face_in, nv_in, mf_in, tf.constant(True))
            onehot_labels = tf.one_hot(labels, depth=NUM_CLASSES)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=onehot_labels)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_metric(y_true=onehot_labels, y_pred=pred_logits)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, meshes, labels):
        vertex_in, face_in, nv_in, mf_in = meshes
        vertex_in = vertex_in[:tf.reduce_sum(nv_in), :]  # change input axis-0 size to None
        face_in = face_in[:tf.reduce_sum(mf_in), :]      # change input axis-0 size to None
        pred_logits = self.model(vertex_in, face_in, nv_in, mf_in,
                                 is_training=tf.constant(False), shuffle_normals=False)
        onehot_labels = tf.one_hot(labels, depth=NUM_CLASSES)
        t_loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=onehot_labels)

        self.test_loss(t_loss)
        self.test_metric(y_true=onehot_labels, y_pred=pred_logits)
        return labels, pred_logits

    def fit(self, train, test, epochs, manager):
        '''
            This fit function runs training and testing.
        '''
        if opt.ckpt_epoch is not None:
            ckpt_epoch = opt.ckpt_epoch
            checkpoint = os.path.dirname(manager.latest_checkpoint) + '/ckpt-%d'%ckpt_epoch
            ckpt.restore(checkpoint)
            ckpt.step = tf.Variable(ckpt_epoch+1,trainable=False)
            print("Restored from {}".format(checkpoint))
            print('checkpoint epoch is %d'%ckpt_epoch, int(ckpt.optimizer.iterations), int(ckpt.step))
        else:
            ckpt.restore(manager.latest_checkpoint)
            print('ckpt.step=%d'%ckpt.step)
            if manager.latest_checkpoint:
                print("Restored from {}".format(manager.latest_checkpoint))
                ckpt_epoch = int(manager.latest_checkpoint.split('-')[-1])
                ckpt.step = tf.Variable(ckpt_epoch+1,trainable=False)
            else:
                print("Initializing from scratch.")
                ckpt_epoch = 0

        train_template = 'training batches %03d, mean loss: %3.4f, accuracy: %3.2f, runtime-per-batch: %3.2f ms'
        test_template = 'epoch %03d, test mean loss: %3.2f, test accuracy: %3.2f, runtime-4-batches: %3.2f ms'

        for epoch in range(ckpt_epoch+1, epochs):
            log_string(' ****************************** EPOCH %03d TRAINING ******************************'%(epoch))
            log_string(str(datetime.now()))
            sys.stdout.flush()

            batch_idx, train_time = 0, 0.0
            for meshes, labels in train.batch(opt.batch_size, drop_remainder=tf.constant(True)):
                now = time.time()
                self.train_step(meshes, labels)
                batch_time = (time.time() - now)
                train_time += batch_time

                batch_idx += 1
                with self.train_writer.as_default():
                    tf.summary.scalar('train_loss: ', self.train_loss.result(),
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('train_Acc: ', self.train_metric.result()*100,
                                      step=int(ckpt.optimizer.iterations))
                    tf.summary.scalar('learning_rate: ', optimizer._decayed_lr(tf.float32),
                                      step=int(ckpt.optimizer.iterations))
                train_writer.flush()
                if batch_idx%50==0:
                    log_string(train_template%(batch_idx, self.train_loss.result(),
                                               self.train_metric.result()*100,
                                               train_time/batch_idx*1000))

            save_path = manager.save(ckpt.step)
            ckpt.step.assign_add(1)
            log_string("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

            log_string(' ------------------------------ EPOCH %03d EVALUATION ------------------------------'%(epoch))
            log_string(str(datetime.now()))
            batch_idx, test_time = 0, 0.0
            for test_meshes, test_labels in test.batch(64, drop_remainder=tf.constant(False)):
                now = time.time()
                self.test_step(test_meshes, test_labels)
                batch_time = (time.time() - now)
                test_time += batch_time
                batch_idx += 1

            with self.test_writer.as_default():
                tf.summary.scalar('test_loss: ', self.test_loss.result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_Acc: ', self.test_metric.result()*100,
                                  step=int(ckpt.optimizer.iterations))
            test_writer.flush()
            log_string(test_template%(epoch, self.test_loss.result(),
                                      self.test_metric.result()*100,
                                      test_time/batch_idx*1000))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_metric.reset_states()
            self.test_loss.reset_states()
            self.test_metric.reset_states()


if __name__=='__main__':
    # load in datasets
    trainSet = MeshDataset(Lists['train']).shuffle(buffer_size=len(Lists['train'])).map(parse_fn, is_training=True)
    valSet  = MeshDataset(Lists['val']).map(parse_fn, is_training=False)
    testSet = MeshDataset(Lists['test']).map(parse_fn, is_training=False)

    # create model & Make a loss object
    picasso_net = PicassoNetII(num_class=NUM_CLASSES, stride=[3,2,2,2], mix_components=opt.num_clusters)

    # Select the adam optimizer
    starter_learning_rate = 0.001
    decay_steps = 20000
    decay_rate = 0.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        starter_learning_rate, decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Specify metrics for training
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    # Specify metrics for testing
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_metric = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    # create log directory for train and loss summary
    logdir_train = os.path.join(LOG_DIR, "train/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    logdir_test = os.path.join(LOG_DIR, "test/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    # Creates a file writer for the log directory.
    train_writer = tf.summary.create_file_writer(logdir_train)
    test_writer = tf.summary.create_file_writer(logdir_test)

    # Create an instance of the model
    model = MyModel(net=picasso_net, optimizer=optimizer,
                    train_loss=train_loss, train_metric=train_metric,
                    test_loss=test_loss, test_metric=test_metric,
                    train_writer=train_writer, test_writer=test_writer)

    # create a checkpoint that will manage all the objects with trackable state
    ckpt = tf.train.Checkpoint(step=tf.Variable(1,trainable=False), optimizer=optimizer,
                               net=picasso_net, train_loss=train_loss, train_metric=train_metric,
                               test_loss=test_loss, test_metric=test_metric)
    manager = tf.train.CheckpointManager(ckpt, LOG_DIR+'/tf_ckpts', max_to_keep=500)
    model.fit(train=trainSet, test=testSet, epochs=500, manager=manager)

