import tensorflow as tf
from termcolor import colored


class Dataset:

    def __init__(self, fileList):
        self.dataset = tf.data.Dataset.from_tensor_slices(fileList)

    def shuffle(self, buffer_size=10000):
        self.dataset = self.dataset.shuffle(buffer_size)
        return self

    def map(self, parse_fn, is_training=True):
        self.parse_fn = parse_fn
        self.is_training = is_training
        self.dataset = tf.data.TFRecordDataset(self.dataset)
        return self

    def batch(self, batch_size, max_num_vertices=1600000, drop_remainder=False):
        face_offset_idx = 0
        num_vertices = 0
        vertex, face, nv, mf, label = [], [], [], [], []
        for raw_record in self.dataset.take(-1):
            data = self.parse_fn(raw_record.numpy(), is_training=self.is_training)

            if data is not None:
                if data[0].shape[0]>max_num_vertices:
                    yield data[:4], data[4]
                elif (num_vertices+data[0].shape[0])>max_num_vertices:
                    vertex = tf.concat(vertex, axis=0)
                    face = tf.concat(face, axis=0)
                    nv = tf.concat(nv, axis=0)
                    mf = tf.concat(mf, axis=0)
                    label = tf.concat(label, axis=0)
                    mesh = (vertex, face, nv, mf)
                    yield mesh, label

                    face_offset_idx = 0
                    num_vertices = 0
                    vertex, face, nv, mf, label = [], [], [], [], []

                    num_vertices += data[0].shape[0]
                    vertex.append(data[0])
                    face.append(data[1] + face_offset_idx)
                    nv.append(data[2])
                    mf.append(data[3])
                    label.append(data[4])
                    face_offset_idx += data[2][0]
                else:
                    num_vertices += data[0].shape[0]
                    vertex.append(data[0])
                    face.append(data[1] + face_offset_idx)
                    nv.append(data[2])
                    mf.append(data[3])
                    label.append(data[4])
                    face_offset_idx += data[2][0]
            if len(nv)==batch_size and num_vertices<max_num_vertices:
                vertex = tf.concat(vertex, axis=0)
                face = tf.concat(face, axis=0)
                nv = tf.concat(nv, axis=0)
                mf = tf.concat(mf, axis=0)
                label = tf.concat(label, axis=0)
                mesh = (vertex, face, nv, mf)
                yield mesh, label

                face_offset_idx = 0
                num_vertices = 0
                vertex, face, nv, mf, label = [], [], [], [], []

        if not drop_remainder and len(nv)>0:
            vertex = tf.concat(vertex, axis=0)
            face = tf.concat(face, axis=0)
            nv = tf.concat(nv, axis=0)
            mf = tf.concat(mf, axis=0)
            label = tf.concat(label, axis=0)
            mesh = (vertex, face, nv, mf)
            yield mesh, label

    def batch_render(self, batch_size, max_num_vertices=1600000, drop_remainder=False):
        face_offset_idx = 0
        num_vertices = 0
        vertex, face, nv, mf, face_texture, baryCoeff, num_texture, label = [], [], [], [], [], [], [], []
        for raw_record in self.dataset.take(-1):
            data = self.parse_fn(raw_record.numpy(), is_training=self.is_training)

            if data is not None:
                if data[0].shape[0]>max_num_vertices:
                    yield data[:-1], data[-1]
                elif (num_vertices+data[0].shape[0])>max_num_vertices:
                    vertex = tf.concat(vertex, axis=0)
                    face = tf.concat(face, axis=0)
                    nv = tf.concat(nv, axis=0)
                    mf = tf.concat(mf, axis=0)
                    face_texture = tf.concat(face_texture, axis=0)
                    baryCoeff = tf.concat(baryCoeff, axis=0)
                    num_texture = tf.concat(num_texture, axis=0)
                    label = tf.concat(label, axis=0)
                    mesh = (vertex, face, nv, mf, face_texture, baryCoeff, num_texture)
                    yield mesh, label

                    face_offset_idx = 0
                    num_vertices = 0
                    vertex, face, nv, mf, face_texture, baryCoeff, num_texture, label = [], [], [], [], [], [], [], []

                    num_vertices += data[0].shape[0]
                    vertex.append(data[0])
                    face.append(data[1] + face_offset_idx)
                    nv.append(data[2])
                    mf.append(data[3])
                    face_texture.append(data[4])
                    baryCoeff.append(data[5])
                    num_texture.append(data[6])
                    label.append(data[7])

                    face_offset_idx += data[2][0]
                else:
                    num_vertices += data[0].shape[0]
                    vertex.append(data[0])
                    face.append(data[1] + face_offset_idx)
                    nv.append(data[2])
                    mf.append(data[3])
                    face_texture.append(data[4])
                    baryCoeff.append(data[5])
                    num_texture.append(data[6])
                    label.append(data[7])

                    face_offset_idx += data[2][0]
            if len(nv)==batch_size and num_vertices<max_num_vertices:
                vertex = tf.concat(vertex, axis=0)
                face = tf.concat(face, axis=0)
                nv = tf.concat(nv, axis=0)
                mf = tf.concat(mf, axis=0)
                face_texture = tf.concat(face_texture, axis=0)
                baryCoeff = tf.concat(baryCoeff, axis=0)
                num_texture = tf.concat(num_texture, axis=0)
                label = tf.concat(label, axis=0)
                mesh = (vertex, face, nv, mf, face_texture, baryCoeff, num_texture)
                yield mesh, label

                face_offset_idx = 0
                num_vertices = 0
                vertex, face, nv, mf, face_texture, baryCoeff, num_texture, label = [], [], [], [], [], [], [], []

        if not drop_remainder and len(nv)>0:
            vertex = tf.concat(vertex, axis=0)
            face = tf.concat(face, axis=0)
            nv = tf.concat(nv, axis=0)
            mf = tf.concat(mf, axis=0)
            face_texture = tf.concat(face_texture, axis=0)
            baryCoeff = tf.concat(baryCoeff, axis=0)
            num_texture = tf.concat(num_texture, axis=0)
            label = tf.concat(label, axis=0)
            mesh = (vertex, face, nv, mf, face_texture, baryCoeff, num_texture)
            yield mesh, label