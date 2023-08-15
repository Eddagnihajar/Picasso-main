import tensorflow as tf

class Dataset:

    def __init__(self, fileList):
        self.dataset = tf.data.TFRecordDataset(fileList)

    def shuffle(self ,buffer_size=10000):
        self.dataset = self.dataset.shuffle(buffer_size)
        return self

    def map(self,func_obj):
        self.map = func_obj
        return self

    def batch(self ,batch_size, max_num_points=1600000, drop_remainder=False):
        pcloud, nv, label = [], [], []
        num_points = 0
        for raw_record in self.dataset.take(-1):
            data = self.map(raw_record.numpy())

            if data is not None:
                if data[0].shape[0]>max_num_points:
                    yield data[:2], data[2]
                elif (num_points+data[0].shape[0])>max_num_points:
                    pcloud = tf.concat(pcloud, axis=0)
                    nv = tf.concat(nv, axis=0)
                    label = tf.concat(label, axis=0)
                    yield (pcloud, nv), label

                    num_points = 0
                    pcloud, nv, label = [], [], []

                    num_points += data[0].shape[0]
                    pcloud.append(data[0])
                    nv.append(data[2])
                    label.append(data[4])
                else:
                    num_points += data[0].shape[0]
                    pcloud.append(data[0])
                    nv.append(data[1])
                    label.append(data[2])
            if len(nv)==batch_size and num_points<max_num_points:
                pcloud = tf.concat(pcloud, axis=0)
                nv = tf.concat(nv, axis=0)
                label = tf.concat(label, axis=0)
                yield (pcloud, nv), label

                num_points = 0
                pcloud, nv, label = [], [], []

        if not drop_remainder and len(nv)>0:
            pcloud = tf.concat(pcloud, axis=0)
            nv = tf.concat(nv, axis=0)
            label = tf.concat(label, axis=0)
            yield (pcloud, nv), label