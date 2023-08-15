#!/usr/bin/env bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++14 -c -o decimate_gpu.cu.o decimate_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o decimate_so.so decimate.cpp \
  decimate_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o mesh_conv3d_gpu.cu.o mesh_conv3d_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o mesh_conv3d_so.so mesh_conv3d.cpp \
  mesh_conv3d_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o mesh_pool3d_gpu.cu.o mesh_pool3d_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o mesh_pool3d_so.so mesh_pool3d.cpp \
  mesh_pool3d_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o mesh_unpool3d_gpu.cu.o mesh_unpool3d_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o mesh_unpool3d_so.so mesh_unpool3d.cpp \
  mesh_unpool3d_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}
