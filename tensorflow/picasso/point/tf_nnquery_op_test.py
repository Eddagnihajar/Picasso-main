import sys, os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import time, timeit
from termcolor import colored
from sklearn.neighbors import NearestNeighbors
from tf_nnquery import build_sphere_neighbor, build_cube_neighbor, build_nearest_neighbor


def check_index(radius, nnsample, database, query):
    B, N = database.shape[0:2]
    M = query.shape[1]

    nn_index = []
    nn_dist = []
    nn_count = np.zeros((B*M),dtype=np.int32)
    delta = np.zeros((3,),dtype=np.float32)

    for i in range(B):
        for j in range(M):
            cnt = 0
            for k in range(N):
                delta[0] = database[i,k,0] - query[i,j,0]
                delta[1] = database[i,k,1] - query[i,j,1]
                delta[2] = database[i,k,2] - query[i,j,2]

                dist = np.linalg.norm(delta)
                if (dist<radius) and (cnt<nnsample):
                    nn_index.append(np.int32([i*M+j, i*N+k]))
                    nn_dist.append(dist)
                    cnt += 1

            nn_count[i*M+j] = cnt
    nn_index = np.stack(nn_index, axis=0)
    nn_dist = np.stack(nn_dist, axis=0)
    print('python compute:', nn_count.shape, nn_index.shape, nn_dist.shape)
    return nn_count, nn_index, nn_dist


def check_index_cube(length, nnsample, gridsize, database, query):
    B, N = database.shape[0:2]
    M = query.shape[1]

    nn_index = []
    nn_count = np.zeros((B*M),dtype=np.int32)
    delta = np.zeros((3,),dtype=np.float32)

    for i in range(B):
        for j in range(M):
            cnt = 0
            for k in range(N):
                delta[0] = database[i,k,0] - query[i,j,0]
                delta[1] = database[i,k,1] - query[i,j,1]
                delta[2] = database[i,k,2] - query[i,j,2]

                if abs(delta[0])<length/2 and abs(delta[1])<length/2 and abs(delta[2])<length/2 and (cnt<nnsample):
                    xId = np.int32((delta[0]+length/2)/(length/gridsize))
                    yId = np.int32((delta[1]+length/2)/(length/gridsize))
                    zId = np.int32((delta[2]+length/2)/(length/gridsize))
                    binidx = xId*gridsize*gridsize + yId*gridsize + zId

                    nn_index.append(np.int32([i*M+j, i*N+k, binidx]))
                    cnt += 1

            nn_count[i*M+j] = cnt
    nn_index = np.stack(nn_index, axis=0)
    print('python compute:', nn_count.shape, nn_index.shape)
    return nn_count, nn_index


def check_index_nn3(database, query):
    nn_index, nn_dist = [], []
    for b in range(B):
        nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(database[b,...])
        distances, indices = nbrs.kneighbors(query[b, ...])
        distances = distances.astype('float32')
        indices = indices.astype('int32')
        nn_dist.append(distances)
        nn_index.append(indices + b * N)
    nn_dist = np.concatenate(nn_dist, axis=0)
    nn_index = np.concatenate(nn_index, axis=0)
    print("python compute", nn_index.shape, nn_dist.shape)
    return nn_index, nn_dist


if __name__=='__main__':
    # tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.set_soft_device_placement(True)
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    np.random.seed(100)
    B, N, M = 128, 16384, 8192 #4, 1000, 1000 #
    radius, nnsample = 0.1, 2147483647
    tmp1 = np.random.random((B,N,3)).astype('float32')
    tmp2 = np.random.random((B,M,3)).astype('float32')
    maxIter = 100
    vecN = np.int32([N]*B)
    vecM = np.int32([M]*B)
    gridSize = 3

    # tensor variables
    database = tf.constant(tmp1)
    query = tf.constant(tmp2)
    database = tf.reshape(database,shape=[-1,3])
    query = tf.reshape(query,shape=[-1,3])
    nvDatabase = tf.constant(vecN)
    nvQuery = tf.constant(vecM)

    # # build_sphere_neighbor
    # cntInfo, nnCount, nnIndex, nnDist = build_sphere_neighbor(database, query, nvDatabase, nvQuery,
    #                                                           radius=radius, nnsample=nnsample)
    # print(tf.reduce_max(nnCount), tf.reduce_mean(tf.cast(nnCount,dtype=tf.float32)))
    #
    # print(cntInfo.numpy()[-10:], '\n', nnCount.numpy()[-10:])
    #
    # check_nn_count, check_nn_index, check_nn_dist = check_index(radius, nnsample, tmp1, tmp2)
    # if np.array_equal(nnIndex, check_nn_index) and \
    #         np.array_equal(nnCount, check_nn_count) and \
    #         np.allclose(nnDist, check_nn_dist):
    #     print(colored('build_sphere_neighbor passed the test!\n', 'green'))
    # else:
    #     print(np.array_equal(nnIndex, check_nn_index), \
    #           np.array_equal(nnCount, check_nn_count), \
    #           np.allclose(nnDist, check_nn_dist))
    #
    #
    # # build_cube_neighbor
    # cntInfo, nnCount, nnIndex = build_cube_neighbor(database, query, nvDatabase, nvQuery, length=2*radius,
    #                                                 nnsample=nnsample, gridsize=gridSize)
    # print(tf.reduce_max(nnCount), tf.reduce_mean(tf.cast(nnCount,dtype=tf.float32)))
    # check_nn_count, check_nn_index = check_index_cube(2*radius, nnsample, gridSize, tmp1, tmp2)
    # if np.array_equal(nnIndex, check_nn_index) and np.array_equal(nnCount, check_nn_count):
    #     print(colored('build_cube_neighbor passed the test!\n', 'green'))
    # else:
    #     print(np.array_equal(nnIndex[..., 0], check_nn_index[..., 0]), \
    #           np.array_equal(nnIndex[..., 1], check_nn_index[..., 1]), \
    #           np.array_equal(nnCount, check_nn_count))
    #     print(np.sum(nnIndex[..., 0] == check_nn_index[..., 0]) / np.sum(nnCount), \
    #           np.sum(nnIndex[..., 1] == check_nn_index[..., 1]) / np.sum(nnCount))
    #     print(np.amax(nnIndex[:, :, :, 1]))
    #     for b in range(B):
    #         A = nnIndex[b, :, :, 1]
    #         B = check_nn_index[b, :, :, 1]
    #         print(A[0, :], B[0, :])
    #         print(np.array_equal(A, B))
    #
    #
    # # build_nearest_neighbor
    # nnIndex, nnDist = build_nearest_neighbor(database, query, nvDatabase, nvQuery)
    # check_nn_index, check_nn_dist = check_index_nn3(tmp1, tmp2)
    # if np.array_equal(nnIndex, check_nn_index) and \
    #         np.allclose(nnDist, check_nn_dist):
    #     print(colored('build_nearest_neighbor passed the test!\n', 'green'))
    # else:
    #     print(np.array_equal(nnIndex, check_nn_index), \
    #           np.allclose(nnDist, check_nn_dist))

    cpp_gpu_time = timeit.timeit(lambda: build_sphere_neighbor(database, query, nvDatabase, nvQuery,
                                                               radius=radius, nnsample=nnsample), number=maxIter)
    print(colored('sphereNN elapsed time: {}\n'.format(cpp_gpu_time / maxIter),'magenta'))

    cpp_gpu_time = timeit.timeit(lambda: build_cube_neighbor(database, query, nvDatabase, nvQuery,
                                 length=2 * radius, nnsample=nnsample, gridsize=gridSize), number=maxIter)
    print(colored('cubeNN elapsed time: {}\n'.format(cpp_gpu_time / maxIter), 'magenta'))

    cpp_gpu_time = timeit.timeit(lambda: build_nearest_neighbor(database, query, nvDatabase, nvQuery), number=maxIter)
    print(colored('NN3 elapsed time: {}\n'.format(cpp_gpu_time / maxIter), 'magenta'))
