#!/usr/bin/env bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++14 -c -o buildkernel_gpu.cu.o buildkernel_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o buildkernel_so.so buildkernel.cpp \
  buildkernel_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o pcloud_conv3d_gpu.cu.o pcloud_conv3d_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o pcloud_conv3d_so.so pcloud_conv3d.cpp \
  pcloud_conv3d_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o nnquery_gpu.cu.o nnquery_gpu.cu \
${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o nnquery_so.so nnquery.cpp \
  nnquery_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
   -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o pcloud_pool3d_gpu.cu.o pcloud_pool3d_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o pcloud_pool3d_so.so pcloud_pool3d.cpp \
  pcloud_pool3d_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
   -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o sample_gpu.cu.o sample_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o sample_so.so sample.cpp \
  sample_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}

nvcc -std=c++14 -c -o pcloud_unpool3d_gpu.cu.o pcloud_unpool3d_gpu.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++14 -shared -o pcloud_unpool3d_so.so pcloud_unpool3d.cpp \
  pcloud_unpool3d_gpu.cu.o ${TF_CFLAGS[@]} -fPIC -I/usr/local/cuda/include \
  -lcudart -L/usr/local/cuda/lib64 ${TF_LFLAGS[@]}