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
from picasso.models.scene_seg_texture import PicassoNetII

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='path to tfrecord data')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='../log_s3dis_render', help='Log dir [default: log_s3dis_render]')
parser.add_argument('--ckpt_epoch', type=int, default=None, help='epoch model to load [default: None]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--max_num_vertices', type=int, default=1500000, help='maximum vertices allowed in a batch')
parser.add_argument('--num_clusters', type=int, default=27, help='number of cluster components')
opt = parser.parse_args()

# set gpu
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[opt.gpu], 'GPU')
tf.config.experimental.set_memory_growth(gpus[opt.gpu], True)

LOG_DIR = opt.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (os.path.basename(__file__), LOG_DIR))     # bkp of train procedure
os.system('cp picasso/models/scene_seg_texture.py %s'%LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(opt)+'\n')

# tfrecord file_lists
Lists = {}
Lists['train'] = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'train_files_fold5.txt'))]
Lists['test']  = [line.rstrip() for line in open(os.path.join(opt.data_dir, 'test_files_fold5.txt'))]
num_repeat = np.ceil((600*opt.batch_size)/len(Lists['train'])).astype('int32')
Lists['train'] = Lists['train']*num_repeat

NUM_CLASSES = 13
classnames = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
              'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def softmax_cross_entropy(pred_logits, onehot_labels):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=onehot_labels)
    if loss.shape == 0:
        loss = 0
    else:
        loss = tf.reduce_mean(loss)  # per-element_wise entropy loss

    loss = tf.where(tf.math.is_nan(loss), 0.0, loss)
    return loss


def augment_fn(vertex, texture):
    prob = 0.5
    augment_xyz = vertex
    augment_rgb = texture

    # augment (xyz) on the fly with spatial transformations
    augment_xyz = Augment.rotate_point_cloud(augment_xyz, upaxis=3, prob=prob)
    augment_xyz = Augment.rotate_perturbation_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.flip_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.random_scale_point_cloud(augment_xyz, prob=prob)
    augment_xyz = Augment.shift_point_cloud(augment_xyz, prob=prob)

    # augment (rgb) color on the fly
    augment_rgb = Augment.clip_color(augment_rgb, prob=prob)
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

    vertex = tf.io.decode_raw(features['xyz_raw'], tf.float32)
    face = tf.io.decode_raw(features['face_raw'], tf.int32)
    face_texture = tf.io.decode_raw(features['face_texture_raw'], tf.float32)
    bary_coeff = tf.io.decode_raw(features['bary_coeff_raw'], tf.float32)
    num_texture = tf.io.decode_raw(features['num_texture_raw'], tf.int32)
    label = tf.io.decode_raw(features['seg_label_raw'], tf.int32)

    vertex = tf.reshape(vertex, [-1,3])
    face = tf.reshape(face, [-1,3])
    face_texture = tf.reshape(face_texture, [-1,3])
    bary_coeff = tf.reshape(bary_coeff, [-1,3])
    num_texture = tf.reshape(num_texture, [-1])
    label = tf.reshape(label, [-1])

    if is_training:  # augment vertex
        vertex, face_texture = augment_fn(vertex, face_texture)

        # random drop out vertices with probability 0.5
        vertex, face, label, \
        face_mask = Augment.random_drop_vertex(vertex, face,
                                               vertex_label=label,
                                               drop_rate=0.15, prob=0.5)
        valid_indices = tf.where(tf.repeat(face_mask,num_texture))[:,0]
        face_texture = tf.gather(face_texture, valid_indices)
        bary_coeff = tf.gather(bary_coeff, valid_indices)
        num_texture = tf.gather(num_texture, tf.where(face_mask)[:,0])


    # print(face_texture.shape, bary_coeff.shape, tf.reduce_sum(num_texture))
    # temp_face_indices = tf.repeat(tf.range(face.shape[0]), num_texture)
    # temp_face = tf.gather(face, temp_face_indices)
    # inpoints = bary_coeff[:,0:1]*tf.gather(vertex,temp_face[:,0]) +\
    #            bary_coeff[:,1:2]*tf.gather(vertex,temp_face[:,1]) +\
    #            bary_coeff[:,2:]*tf.gather(vertex,temp_face[:,2])
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(vertex.numpy())
    # mesh.triangles = o3d.utility.Vector3iVector(face.numpy())
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(inpoints.numpy())
    # cloud.colors = o3d.utility.Vector3dVector(face_texture.numpy())
    # o3d.visualization.draw_geometries([cloud], mesh_show_back_face=True)
    # o3d.visualization.draw_geometries([mesh, cloud], mesh_show_back_face=True)

    num_labelled = tf.reduce_sum(tf.cast((label>=0) & (label<NUM_CLASSES),dtype=tf.int32))
    nv, mf = tf.shape(vertex[:,0]), tf.shape(face[:,0])
    ratio = num_labelled/nv[0]
    if is_training and (ratio<0.2):
        return None
    else:
        return vertex, face, nv, mf, face_texture, bary_coeff, num_texture, label
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
    def train_step(self, meshes, seg_labels):
        vertex_in, face_in, nv_in, mf_in, face_texture, bary_coeff, num_texture = meshes
        vertex_in = vertex_in[:tf.reduce_sum(nv_in),:] # change input axis-0 size to None
        face_in = face_in[:tf.reduce_sum(mf_in),:]     # change input axis-0 size to None
        face_texture = face_texture[:tf.reduce_sum(num_texture),:]
        bary_coeff = bary_coeff[:tf.reduce_sum(num_texture),:]
        num_texture = num_texture[:tf.reduce_sum(mf_in)]
        with tf.GradientTape() as tape:
            pred_logits = self.model(vertex_in, face_in, nv_in, mf_in,
                                     face_texture, bary_coeff, num_texture, True, True)
            indices = tf.where(seg_labels>=0)  # the considered indices
            pred_logits = tf.gather_nd(pred_logits, indices)
            seg_labels = tf.gather_nd(seg_labels, indices)
            onehot_labels = tf.one_hot(seg_labels, depth=NUM_CLASSES)
            loss = softmax_cross_entropy(pred_logits, onehot_labels)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)
        self.train_metric(y_true=onehot_labels, y_pred=pred_logits)

    @tf.function(experimental_relax_shapes=True)
    def test_step(self, meshes, seg_labels):
        vertex_in, face_in, nv_in, mf_in, face_texture, bary_coeff, num_texture = meshes
        pred_logits = self.model(vertex_in, face_in, nv_in, mf_in,
                                 face_texture, bary_coeff, num_texture, False, False)
        indices = tf.where(seg_labels>=0)  # the considered indices
        pred_logits = tf.gather_nd(pred_logits, indices)
        seg_labels = tf.gather_nd(seg_labels, indices)
        onehot_labels = tf.one_hot(seg_labels, depth=NUM_CLASSES)
        t_loss = softmax_cross_entropy(pred_logits, onehot_labels)

        self.test_loss(t_loss)
        self.test_metric(y_true=onehot_labels, y_pred=pred_logits)
        return seg_labels, pred_logits

    @staticmethod
    def evaluate_iou(gt_labels, pred_labels):
        class_names = classnames[:NUM_CLASSES]
        total_seen_class = {cat: 0 for cat in class_names}
        total_correct_class = {cat: 0 for cat in class_names}
        total_union_class = {cat: 0 for cat in class_names}

        for l, cat in enumerate(class_names):
            total_seen_class[cat] += np.sum(gt_labels==l)
            total_union_class[cat] += (np.sum((pred_labels==l) | (gt_labels==l)))
            total_correct_class[cat] += (np.sum((pred_labels==l) & (gt_labels==l)))

        class_iou = {cat: 0.0 for cat in class_names}
        class_acc = {cat: 0.0 for cat in class_names}
        for cat in class_names:
            class_iou[cat] = total_correct_class[cat]/(float(total_union_class[cat]) + np.finfo(float).eps)
            class_acc[cat] = total_correct_class[cat]/(float(total_seen_class[cat]) + np.finfo(float).eps)

        total_correct = sum(list(total_correct_class.values()))
        total_seen = sum(list(total_seen_class.values()))
        log_string('eval overall class accuracy:\t %d/%d=%3.2f'%(total_correct, total_seen,
                                                                 100*total_correct/float(total_seen)))
        log_string('eval average class accuracy:\t %3.2f'%(100*np.mean(list(class_acc.values()))))
        for cat in class_names:
            log_string('eval mIoU of %14s:\t %3.2f'%(cat, 100*class_iou[cat]))
        log_string('eval mIoU of all %d classes:\t %3.2f'%(NUM_CLASSES, 100*np.mean(list(class_iou.values()))))

    def fit(self, train, test, epochs, manager):
        '''
            This fit function runs training and testing.
        '''
        if opt.ckpt_epoch is not None:
            ckpt_epoch = opt.ckpt_epoch
            checkpoint = os.path.dirname(manager.latest_checkpoint) + '/ckpt-%d'%ckpt_epoch
            ckpt.restore(checkpoint)
            ckpt.step = tf.Variable(ckpt_epoch + 1, trainable=False)
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
        test_template = 'epoch %03d, test mean loss: %3.2f, test accuracy: %3.2f, runtime-per-batches: %3.2f ms'

        for epoch in range(ckpt_epoch+1, epochs):
            log_string(' ****************************** EPOCH %03d TRAINING ******************************'%(epoch))
            log_string(str(datetime.now()))
            sys.stdout.flush()

            batch_idx, train_time = 0, 0.0
            for meshes, labels in train.batch_render(opt.batch_size, max_num_vertices=opt.max_num_vertices,
                                                     drop_remainder=tf.constant(True)):
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
            all_gt_labels, all_pred_labels = [], []
            for test_meshes, test_labels in test.batch_render(1, drop_remainder=tf.constant(False)):
                now = time.time()
                gt_labels, pred_logits = self.test_step(test_meshes, test_labels)
                batch_time = (time.time() - now)
                test_time += batch_time
                batch_idx += 1
                pred_labels = tf.math.argmax(pred_logits, axis=-1)
                all_gt_labels.append(gt_labels.numpy())
                all_pred_labels.append(pred_labels.numpy())
            all_gt_labels = np.concatenate(all_gt_labels, axis=0)
            all_pred_labels = np.concatenate(all_pred_labels, axis=0)

            with self.test_writer.as_default():
                tf.summary.scalar('test_loss: ', self.test_loss.result(),
                                  step=int(ckpt.optimizer.iterations))
                tf.summary.scalar('test_Acc: ', self.test_metric.result()*100,
                                  step=int(ckpt.optimizer.iterations))
            test_writer.flush()
            log_string(test_template%(epoch, self.test_loss.result(),
                                      self.test_metric.result()*100,
                                      test_time/batch_idx*1000))
            self.evaluate_iou(all_gt_labels, all_pred_labels)

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_metric.reset_states()
            self.test_loss.reset_states()
            self.test_metric.reset_states()


if __name__=='__main__':

    # load in datasets
    trainSet = MeshDataset(Lists['train']).shuffle(buffer_size=len(Lists['train'])).map(parse_fn, is_training=True)
    testSet = MeshDataset(Lists['test']).map(parse_fn, is_training=False)

    # create model & Make a loss object
    picasso_net = PicassoNetII(num_class=NUM_CLASSES, mix_components=opt.num_clusters, use_height=True)

    # Select the adam optimizer
    starter_learning_rate = 0.001
    decay_steps = 20000
    decay_rate = 0.5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                  starter_learning_rate, decay_steps, decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer = tfa.optimizers.AdamW(weight_decay=1e-4, learning_rate=lr_schedule)

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
                    test_loss=test_loss,   test_metric=test_metric,
                    train_writer=train_writer, test_writer=test_writer)

    # create a checkpoint that will manage all the objects with trackable state
    ckpt = tf.train.Checkpoint(step=tf.Variable(1,trainable=False), optimizer=optimizer,
                               net=picasso_net, train_loss=train_loss, train_metric=train_metric,
                               test_loss=test_loss, test_metric=test_metric)
    manager = tf.train.CheckpointManager(ckpt, LOG_DIR + '/tf_ckpts', max_to_keep=300)
    model.fit(train=trainSet, test=testSet, epochs=300, manager=manager)
