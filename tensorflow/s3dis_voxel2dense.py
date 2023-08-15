import os, argparse
import numpy as np
import open3d as o3d
import glob
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument('--dense_dir', required=True, help='path to the raw dense point cloud data')
parser.add_argument('--voxel_dir', default='../log_s3dis_plain/test_results',
                    help='Log dir [default: ../log_s3dis_plain/test_results]')
opt = parser.parse_args()

denseFolder = opt.dense_dir
voxelFolder = opt.voxel_dir
sceneList = [os.path.basename(x)[7:-4] for x in glob.glob(os.path.join(voxelFolder,'Vertex/*.txt'))]

NUM_CLASSES = 13
labels = [*range(NUM_CLASSES)]
class_names = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']
total_seen_class = {cat: 0 for cat in class_names}
total_correct_class = {cat: 0 for cat in class_names}
total_union_class = {cat: 0 for cat in class_names}
print(sceneList)

for iter, scene_name in enumerate(sceneList):
    denseCloud = np.loadtxt(os.path.join(denseFolder, 'clouds', scene_name+'.txt'), delimiter=',', dtype=np.float32)
    denseLabel = np.loadtxt(os.path.join(denseFolder, 'labels', scene_name+'.txt'), dtype=np.int32)

    center = (np.amin(denseCloud[:,:3],axis=0)+np.amax(denseCloud[:,:3],axis=0))/2
    center[-1] = np.amin(denseCloud[:,3])
    print('orgin of dense cloud: ', center)
    denseCloud[:,:3] = denseCloud[:,:3] - center

    voxelCloud  = np.loadtxt(os.path.join(voxelFolder, 'Vertex/Area_5_%s.txt'%scene_name), dtype=np.float32, delimiter=',')
    xyz = np.asarray(voxelCloud[:,:3], dtype=np.float32)
    rgb = np.asarray(voxelCloud[:,3:], dtype=np.float32)
    voxelPred = np.loadtxt(os.path.join(voxelFolder, 'Pred/Area_5_'+scene_name+'.txt'), dtype=np.int32)
    center = (np.amin(xyz, axis=0)+np.amax(xyz, axis=0))/2

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(xyz)
    distances, indices = nbrs.kneighbors(denseCloud[:,:3])
    densePred = voxelPred[indices[:,0]]

    gt_labels = denseLabel
    pred_labels = densePred
    for l, cat in enumerate(class_names):
        total_seen_class[cat] += np.sum(gt_labels==l)
        total_union_class[cat] += (np.sum((pred_labels==l) | (gt_labels==l)))
        total_correct_class[cat] += (np.sum((pred_labels==l) & (gt_labels==l)))
    print('%d/%d processed, %s'%(iter+1,len(sceneList), scene_name))

class_iou = {cat: 0.0 for cat in class_names}
class_acc = {cat: 0.0 for cat in class_names}
for cat in class_names:
    class_iou[cat] = total_correct_class[cat]/(float(total_union_class[cat]) + np.finfo(float).eps)
    class_acc[cat] = total_correct_class[cat]/(float(total_seen_class[cat]) + np.finfo(float).eps)

total_correct = sum(list(total_correct_class.values()))
total_seen = sum(list(total_seen_class.values()))
print('total number of test sample:\t %d'%len(sceneList))
print('eval overall class accuracy:\t %d/%d=%3.2f'%(total_correct, total_seen,
                                                    100 * total_correct/float(total_seen)))
print('eval average class accuracy:\t %3.2f'%(100*np.mean(list(class_acc.values()))))
for cat in class_names:
    print('eval mIoU of %14s:\t %3.2f'%(cat, 100*class_iou[cat]))
print('eval mIoU of all 13 classes:\t %3.2f'%(100*np.mean(list(class_iou.values()))))
